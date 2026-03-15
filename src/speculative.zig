// Speculative Decoding.
//
// The key insight: use a small/fast "draft" model to generate k candidate
// tokens, then verify all k in a single forward pass of the larger model.
// When the draft model is good, most tokens are accepted — giving the
// speed of the small model with the quality of the large one.
//
// Algorithm:
//   1. Draft model generates k tokens autoregressively (fast)
//   2. Verify model scores all k tokens in one batched forward pass
//   3. Accept the longest prefix where draft and verify agree
//   4. For the first disagreement, rejection-sample from the verify distribution
//   5. On average, we accept ~k-1 tokens per verify pass (when draft is good)
//
// Reference: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"

const std = @import("std");
const simd = @import("simd.zig");
const tensor = @import("tensor.zig");
const attention = @import("attention.zig");
const transformer = @import("transformer.zig");

/// Speculative decoding engine.
///
/// DraftModelDim:   embedding dimension for the draft model
/// DraftNumHeads:   number of attention heads in draft model
/// DraftLayers:     number of layers in draft model
/// VerifyModelDim:  embedding dimension for the verify model
/// VerifyNumHeads:  number of attention heads in verify model
/// VerifyLayers:    number of layers in verify model
/// VocabSize:       shared vocabulary size
/// MaxSeqLen:       maximum sequence length
/// MaxDraftTokens:  maximum number of draft tokens per speculation step
pub fn SpeculativeDecoder(
    comptime DraftModelDim: usize,
    comptime DraftNumHeads: usize,
    comptime DraftLayers: usize,
    comptime VerifyModelDim: usize,
    comptime VerifyNumHeads: usize,
    comptime VerifyLayers: usize,
    comptime VocabSize: usize,
    comptime MaxSeqLen: usize,
    comptime MaxDraftTokens: usize,
) type {
    const DraftModel = transformer.Transformer(DraftModelDim, DraftNumHeads, DraftLayers, VocabSize, MaxSeqLen);
    const VerifyModel = transformer.Transformer(VerifyModelDim, VerifyNumHeads, VerifyLayers, VocabSize, MaxSeqLen);

    return struct {
        draft: DraftModel,
        verify: VerifyModel,

        const Self = @This();
        pub const draft_model_dim = DraftModelDim;
        pub const verify_model_dim = VerifyModelDim;
        pub const draft_layers = DraftLayers;
        pub const verify_layers = VerifyLayers;
        pub const vocab_size = VocabSize;
        pub const max_draft_tokens = MaxDraftTokens;
        pub const max_seq_len = MaxSeqLen;

        pub fn init() Self {
            return .{
                .draft = DraftModel.init(),
                .verify = VerifyModel.init(),
            };
        }

        /// Result of one speculation step.
        pub const SpeculationResult = struct {
            /// Tokens accepted (including rejection-sampled token)
            tokens: [MaxDraftTokens + 1]usize,
            /// Number of tokens accepted
            accepted: usize,
            /// Number of draft tokens generated
            drafted: usize,
        };

        /// Compute softmax probabilities from logits.
        fn softmax(logits: *const [VocabSize]f32) [VocabSize]f32 {
            var max_val: f32 = logits[0];
            for (1..VocabSize) |i| {
                if (logits[i] > max_val) max_val = logits[i];
            }
            var probs: [VocabSize]f32 = undefined;
            var sum_exp: f32 = 0;
            for (0..VocabSize) |i| {
                probs[i] = @exp(logits[i] - max_val);
                sum_exp += probs[i];
            }
            const inv = 1.0 / sum_exp;
            for (0..VocabSize) |i| probs[i] *= inv;
            return probs;
        }

        /// Sample from a probability distribution.
        fn sampleFromProbs(probs: *const [VocabSize]f32, rng: *std.Random) usize {
            var r = rng.float(f32);
            for (0..VocabSize) |i| {
                r -= probs[i];
                if (r <= 0) return i;
            }
            return VocabSize - 1;
        }

        /// Rejection sampling: sample from max(0, p_verify - p_draft), normalized.
        fn rejectionSample(
            verify_probs: *const [VocabSize]f32,
            draft_probs: *const [VocabSize]f32,
            rng: *std.Random,
        ) usize {
            var adjusted: [VocabSize]f32 = undefined;
            var total: f32 = 0;
            for (0..VocabSize) |i| {
                adjusted[i] = @max(verify_probs[i] - draft_probs[i], 0.0);
                total += adjusted[i];
            }
            if (total <= 1e-10) {
                // Distributions match perfectly — sample from verify
                return sampleFromProbs(verify_probs, rng);
            }
            const inv = 1.0 / total;
            for (0..VocabSize) |i| adjusted[i] *= inv;
            return sampleFromProbs(&adjusted, rng);
        }

        /// Run one speculative decoding step.
        ///
        /// Generates up to MaxDraftTokens candidates with the draft model,
        /// then verifies them with the full model.
        ///
        /// Returns the accepted tokens and count.
        pub fn speculate(
            self: *const Self,
            last_token: usize,
            draft_caches: *[DraftLayers]attention.KVCache(DraftModelDim, MaxSeqLen),
            verify_caches: *[VerifyLayers]attention.KVCache(VerifyModelDim, MaxSeqLen),
            pos: usize,
            rng: *std.Random,
        ) SpeculationResult {
            var result: SpeculationResult = .{
                .tokens = [_]usize{0} ** (MaxDraftTokens + 1),
                .accepted = 0,
                .drafted = 0,
            };

            // Step 1: Draft model generates k candidate tokens
            var draft_tokens: [MaxDraftTokens]usize = undefined;
            var draft_logits: [MaxDraftTokens][VocabSize]f32 = undefined;
            var cur_token = last_token;

            for (0..MaxDraftTokens) |i| {
                const cur_pos = pos + i;
                if (cur_pos >= MaxSeqLen) break;

                const logits = self.draft.forward(cur_token, draft_caches, cur_pos);
                draft_logits[i] = logits;

                // Greedy draft for simplicity
                cur_token = attention.greedySample(VocabSize, &logits);
                draft_tokens[i] = cur_token;
                result.drafted += 1;
            }

            if (result.drafted == 0) return result;

            // Step 2: Verify model scores all drafted tokens
            // First, score the original last_token position
            var verify_logits: [MaxDraftTokens + 1][VocabSize]f32 = undefined;
            verify_logits[0] = self.verify.forward(last_token, verify_caches, pos);

            // Then score each drafted token
            for (0..result.drafted) |i| {
                const verify_pos = pos + 1 + i;
                if (verify_pos >= MaxSeqLen) break;
                verify_logits[i + 1] = self.verify.forward(draft_tokens[i], verify_caches, verify_pos);
            }

            // Step 3: Accept/reject each drafted token
            for (0..result.drafted) |i| {
                const draft_probs = softmax(&draft_logits[i]);
                const verify_probs = softmax(&verify_logits[i]);

                const drafted_token = draft_tokens[i];
                const p_draft = draft_probs[drafted_token];
                const p_verify = verify_probs[drafted_token];

                // Accept if verify probability >= draft probability
                if (p_verify >= p_draft) {
                    result.tokens[result.accepted] = drafted_token;
                    result.accepted += 1;
                } else {
                    // Probabilistic acceptance: accept with prob p_verify/p_draft
                    const accept_prob = p_verify / (p_draft + 1e-10);
                    if (rng.float(f32) < accept_prob) {
                        result.tokens[result.accepted] = drafted_token;
                        result.accepted += 1;
                    } else {
                        // Rejection: sample from adjusted distribution
                        const adjusted_token = rejectionSample(&verify_probs, &draft_probs, rng);
                        result.tokens[result.accepted] = adjusted_token;
                        result.accepted += 1;
                        break; // Stop after first rejection
                    }
                }
            }

            // If all drafted tokens accepted, sample one more from verify
            if (result.accepted == result.drafted and result.drafted < MaxDraftTokens + 1) {
                const verify_probs = softmax(&verify_logits[result.drafted]);
                result.tokens[result.accepted] = sampleFromProbs(&verify_probs, rng);
                result.accepted += 1;
            }

            return result;
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "speculative decoder init" {
    const SD = SpeculativeDecoder(16, 2, 1, 32, 4, 2, 64, 16, 3);
    try std.testing.expectEqual(@as(usize, 16), SD.draft_model_dim);
    try std.testing.expectEqual(@as(usize, 32), SD.verify_model_dim);
    try std.testing.expectEqual(@as(usize, 1), SD.draft_layers);
    try std.testing.expectEqual(@as(usize, 2), SD.verify_layers);
    try std.testing.expectEqual(@as(usize, 64), SD.vocab_size);
    try std.testing.expectEqual(@as(usize, 3), SD.max_draft_tokens);
}

test "softmax produces valid probabilities" {
    const SD = SpeculativeDecoder(8, 1, 1, 8, 1, 1, 8, 8, 2);
    const logits = [8]f32{ 1.0, 2.0, 3.0, 0.5, -1.0, 0.0, 1.5, 2.5 };
    const probs = SD.softmax(&logits);

    // Should sum to 1
    var total: f32 = 0;
    for (probs) |p| {
        try std.testing.expect(p >= 0);
        total += p;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    // Highest logit should have highest probability
    try std.testing.expect(probs[2] > probs[0]);
    try std.testing.expect(probs[2] > probs[4]);
}

test "rejection sample produces valid token" {
    const SD = SpeculativeDecoder(8, 1, 1, 8, 1, 1, 8, 8, 2);
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var verify_probs = [8]f32{ 0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01 };
    var draft_probs = [8]f32{ 0.1, 0.1, 0.5, 0.1, 0.05, 0.05, 0.05, 0.05 };

    // Sample many times and verify all tokens are in range
    for (0..100) |_| {
        const token = SD.rejectionSample(&verify_probs, &draft_probs, &rng);
        try std.testing.expect(token < 8);
    }
}

test "speculative decoding produces accepted tokens" {
    const SD = SpeculativeDecoder(8, 1, 1, 8, 1, 1, 16, 16, 3);
    const sd = SD.init();

    var draft_caches: [1]attention.KVCache(8, 16) = .{attention.KVCache(8, 16).init()};
    var verify_caches: [1]attention.KVCache(8, 16) = .{attention.KVCache(8, 16).init()};

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const result = sd.speculate(0, &draft_caches, &verify_caches, 0, &rng);

    // Should accept at least 1 token
    try std.testing.expect(result.accepted >= 1);
    try std.testing.expect(result.accepted <= SD.max_draft_tokens + 1);
    try std.testing.expect(result.drafted > 0);

    // All accepted tokens should be valid vocab indices
    for (0..result.accepted) |i| {
        try std.testing.expect(result.tokens[i] < SD.vocab_size);
    }
}
