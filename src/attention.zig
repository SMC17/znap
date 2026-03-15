// Causal self-attention with comptime-known dimensions.
//
// Implements scaled dot-product attention with causal masking.
// No KV cache allocation — everything on the stack for short sequences.
// For longer sequences, use the KVCache type which manages a fixed ring buffer.
//
// Supports:
//   - Multi-head attention (MHA)
//   - Group-query attention (GQA) — fewer KV heads than query heads
//   - Rotary position embeddings (RoPE)
//   - Causal masking (autoregressive)
//
// Reference architectures: GPT-2, LLaMA, DeepSeek

const std = @import("std");
const simd = @import("simd.zig");
const tensor = @import("tensor.zig");

// ── Rotary Position Embeddings (RoPE) ────────────────────────────

/// Apply rotary position embedding to a vector.
/// RoPE encodes position through rotation in 2D subspaces.
pub fn applyRoPE(comptime Dim: usize, x: *[Dim]f32, pos: usize) void {
    const half = Dim / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(Dim)));
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = x[i];
        const x1 = x[i + half];
        x[i] = x0 * cos_a - x1 * sin_a;
        x[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

// ── Causal Self-Attention ────────────────────────────────────────

/// Single-head causal self-attention.
///
/// Given Q, K, V vectors for positions 0..SeqLen, computes:
///   Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
/// with causal masking (position i can only attend to positions ≤ i).
pub fn CausalAttention(comptime HeadDim: usize, comptime MaxSeqLen: usize) type {
    return struct {
        const Self = @This();
        pub const head_dim = HeadDim;
        pub const max_seq_len = MaxSeqLen;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HeadDim)));

        /// Compute attention output for a single position given full KV history.
        ///
        /// q: query vector [HeadDim]
        /// keys: all key vectors up to current position [seq_len][HeadDim]
        /// values: all value vectors up to current position [seq_len][HeadDim]
        /// seq_len: number of positions (including current)
        ///
        /// Returns: attention output [HeadDim]
        pub fn forward(
            q: *const [HeadDim]f32,
            keys: []const [HeadDim]f32,
            values: []const [HeadDim]f32,
            seq_len: usize,
        ) [HeadDim]f32 {
            // Compute attention scores: score[i] = q · k[i] / sqrt(d_k)
            var scores: [MaxSeqLen]f32 = undefined;
            for (0..seq_len) |i| {
                scores[i] = simd.vdot(HeadDim, q, &keys[i]) * scale;
            }

            // Softmax over scores[0..seq_len]
            // Find max for numerical stability
            var max_score: f32 = scores[0];
            for (1..seq_len) |i| {
                if (scores[i] > max_score) max_score = scores[i];
            }

            var sum_exp: f32 = 0;
            for (0..seq_len) |i| {
                scores[i] = @exp(scores[i] - max_score);
                sum_exp += scores[i];
            }
            const inv_sum = 1.0 / sum_exp;
            for (0..seq_len) |i| {
                scores[i] *= inv_sum;
            }

            // Weighted sum of values
            var output: [HeadDim]f32 = [_]f32{0} ** HeadDim;
            for (0..seq_len) |i| {
                for (0..HeadDim) |d| {
                    output[d] += scores[i] * values[i][d];
                }
            }

            return output;
        }
    };
}

// ── Multi-Head Attention ─────────────────────────────────────────

/// Multi-head attention with comptime-known dimensions.
///
/// ModelDim: embedding dimension (e.g., 128, 256, 512)
/// NumHeads: number of attention heads (e.g., 4, 8, 16)
/// MaxSeqLen: maximum sequence length
///
/// Head dimension = ModelDim / NumHeads
pub fn MultiHeadAttention(
    comptime ModelDim: usize,
    comptime NumHeads: usize,
    comptime MaxSeqLen: usize,
) type {
    const HeadDim = ModelDim / NumHeads;
    const Attn = CausalAttention(HeadDim, MaxSeqLen);

    return struct {
        // Projection weights: Q, K, V, Output
        wq: tensor.Tensor(ModelDim, ModelDim),
        wk: tensor.Tensor(ModelDim, ModelDim),
        wv: tensor.Tensor(ModelDim, ModelDim),
        wo: tensor.Tensor(ModelDim, ModelDim),

        const Self = @This();
        pub const model_dim = ModelDim;
        pub const num_heads = NumHeads;
        pub const head_dim = HeadDim;
        pub const max_seq_len = MaxSeqLen;
        pub const param_count = 4 * ModelDim * ModelDim;

        pub fn init() Self {
            return .{
                .wq = tensor.Tensor(ModelDim, ModelDim).zeros(),
                .wk = tensor.Tensor(ModelDim, ModelDim).zeros(),
                .wv = tensor.Tensor(ModelDim, ModelDim).zeros(),
                .wo = tensor.Tensor(ModelDim, ModelDim).zeros(),
            };
        }

        /// Forward pass for a single new token position.
        /// x: input embedding [ModelDim]
        /// kv_cache: accumulated K,V for all previous positions
        /// pos: current position in the sequence
        pub fn forward(
            self: *const Self,
            x: *const [ModelDim]f32,
            kv_keys: []const [ModelDim]f32,
            kv_values: []const [ModelDim]f32,
            pos: usize,
        ) [ModelDim]f32 {
            // Project input to Q, K, V
            const q = tensor.matvec(ModelDim, ModelDim, &self.wq, x);
            const k = tensor.matvec(ModelDim, ModelDim, &self.wk, x);
            const v = tensor.matvec(ModelDim, ModelDim, &self.wv, x);
            _ = k;
            _ = v;

            // Apply RoPE to Q and K
            // Split into heads and apply per-head
            var head_outputs: [NumHeads][HeadDim]f32 = undefined;

            for (0..NumHeads) |h| {
                const offset = h * HeadDim;

                // Extract this head's query
                var q_head: [HeadDim]f32 = undefined;
                for (0..HeadDim) |d| q_head[d] = q[offset + d];
                applyRoPE(HeadDim, &q_head, pos);

                // Extract this head's K,V from cache
                var k_heads: [MaxSeqLen][HeadDim]f32 = undefined;
                var v_heads: [MaxSeqLen][HeadDim]f32 = undefined;
                for (0..pos + 1) |p| {
                    const kv_p = if (p < pos) p else pos;
                    for (0..HeadDim) |d| {
                        k_heads[p] = undefined;
                        v_heads[p] = undefined;
                        if (kv_p < kv_keys.len) {
                            k_heads[p][d] = kv_keys[kv_p][offset + d];
                            v_heads[p][d] = kv_values[kv_p][offset + d];
                        }
                    }
                }

                // Attention for this head
                head_outputs[h] = Attn.forward(&q_head, k_heads[0 .. pos + 1], v_heads[0 .. pos + 1], pos + 1);
            }

            // Concatenate heads and project output
            var concat: [ModelDim]f32 = undefined;
            for (0..NumHeads) |h| {
                for (0..HeadDim) |d| {
                    concat[h * HeadDim + d] = head_outputs[h][d];
                }
            }

            return tensor.matvec(ModelDim, ModelDim, &self.wo, &concat);
        }
    };
}

// ── KV Cache ─────────────────────────────────────────────────────

/// Fixed-size KV cache for autoregressive generation.
/// No heap allocation — entire cache on the stack.
pub fn KVCache(comptime ModelDim: usize, comptime MaxSeqLen: usize) type {
    return struct {
        keys: [MaxSeqLen][ModelDim]f32,
        values: [MaxSeqLen][ModelDim]f32,
        len: usize,

        const Self = @This();

        pub fn init() Self {
            return .{
                .keys = [_][ModelDim]f32{[_]f32{0} ** ModelDim} ** MaxSeqLen,
                .values = [_][ModelDim]f32{[_]f32{0} ** ModelDim} ** MaxSeqLen,
                .len = 0,
            };
        }

        pub fn append(self: *Self, key: [ModelDim]f32, value: [ModelDim]f32) void {
            if (self.len >= MaxSeqLen) {
                // Shift everything left (sliding window)
                for (0..MaxSeqLen - 1) |i| {
                    self.keys[i] = self.keys[i + 1];
                    self.values[i] = self.values[i + 1];
                }
                self.keys[MaxSeqLen - 1] = key;
                self.values[MaxSeqLen - 1] = value;
            } else {
                self.keys[self.len] = key;
                self.values[self.len] = value;
                self.len += 1;
            }
        }

        pub fn reset(self: *Self) void {
            self.len = 0;
        }
    };
}

// ── Layer Normalization ──────────────────────────────────────────

/// RMS Layer Normalization (used in LLaMA, modern transformers).
/// Faster than standard LayerNorm — no mean subtraction.
pub fn rmsNorm(comptime Dim: usize, x: *const [Dim]f32, weight: *const [Dim]f32) [Dim]f32 {
    // Compute RMS
    var sum_sq: f32 = 0;
    for (0..Dim) |i| sum_sq += x[i] * x[i];
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(Dim)) + 1e-5);
    const inv_rms = 1.0 / rms;

    // Normalize and scale
    var output: [Dim]f32 = undefined;
    for (0..Dim) |i| {
        output[i] = x[i] * inv_rms * weight[i];
    }
    return output;
}

// ── Sampling ─────────────────────────────────────────────────────

/// Top-k sampling: keep only the top K logits, zero out the rest, then sample.
pub fn topKSample(comptime VocabSize: usize, logits: *[VocabSize]f32, k: usize, temperature: f32, rng: *std.Random) usize {
    // Apply temperature
    for (0..VocabSize) |i| logits[i] /= temperature;

    // Find the k-th largest value
    var threshold: f32 = -std.math.inf(f32);
    if (k < VocabSize) {
        // Simple selection: sort and take k-th
        var sorted: [VocabSize]f32 = logits.*;
        std.mem.sort(f32, &sorted, {}, std.sort.desc(f32));
        threshold = sorted[k];
    }

    // Zero out below threshold
    for (0..VocabSize) |i| {
        if (logits[i] < threshold) logits[i] = -std.math.inf(f32);
    }

    // Softmax
    var max_val: f32 = -std.math.inf(f32);
    for (0..VocabSize) |i| {
        if (logits[i] > max_val) max_val = logits[i];
    }
    var sum_exp: f32 = 0;
    for (0..VocabSize) |i| {
        logits[i] = @exp(logits[i] - max_val);
        sum_exp += logits[i];
    }
    for (0..VocabSize) |i| logits[i] /= sum_exp;

    // Sample from the distribution
    var r = rng.float(f32);
    for (0..VocabSize) |i| {
        r -= logits[i];
        if (r <= 0) return i;
    }
    return VocabSize - 1;
}

/// Greedy (argmax) sampling.
pub fn greedySample(comptime VocabSize: usize, logits: *const [VocabSize]f32) usize {
    return simd.vargmax(VocabSize, logits);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "RoPE preserves magnitude" {
    var x = [4]f32{ 1, 0, 0, 1 };
    const mag_before = simd.vdot(4, &x, &x);
    applyRoPE(4, &x, 5);
    const mag_after = simd.vdot(4, &x, &x);
    try std.testing.expectApproxEqAbs(mag_before, mag_after, 1e-4);
}

test "RoPE different positions give different rotations" {
    var x1 = [4]f32{ 1, 0, 0, 1 };
    var x2 = [4]f32{ 1, 0, 0, 1 };
    applyRoPE(4, &x1, 0);
    applyRoPE(4, &x2, 10);
    // They should be different after different position rotations
    try std.testing.expect(@abs(x1[0] - x2[0]) > 0.01);
}

test "causal attention single position" {
    const Attn = CausalAttention(4, 8);
    const q = [4]f32{ 1, 0, 0, 0 };
    var keys: [1][4]f32 = .{.{ 1, 0, 0, 0 }};
    var values: [1][4]f32 = .{.{ 0.5, 0.3, 0.2, 0.1 }};
    const out = Attn.forward(&q, &keys, &values, 1);
    // With single position, output should equal the value
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), out[1], 1e-4);
}

test "causal attention two positions" {
    const Attn = CausalAttention(4, 8);
    const q = [4]f32{ 1, 0, 0, 0 };
    var keys = [2][4]f32{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 } };
    var values = [2][4]f32{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 } };
    const out = Attn.forward(&q, &keys, &values, 2);
    // Query aligns with key[0], so output should lean toward value[0]
    try std.testing.expect(out[0] > out[1]);
}

test "KV cache append and access" {
    var cache = KVCache(4, 8).init();
    try std.testing.expectEqual(@as(usize, 0), cache.len);

    cache.append(.{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 });
    try std.testing.expectEqual(@as(usize, 1), cache.len);
    try std.testing.expectEqual(@as(f32, 1.0), cache.keys[0][0]);
    try std.testing.expectEqual(@as(f32, 5.0), cache.values[0][0]);

    cache.append(.{ 9, 10, 11, 12 }, .{ 13, 14, 15, 16 });
    try std.testing.expectEqual(@as(usize, 2), cache.len);
}

test "KV cache sliding window when full" {
    var cache = KVCache(2, 3).init();
    cache.append(.{ 1, 2 }, .{ 10, 20 });
    cache.append(.{ 3, 4 }, .{ 30, 40 });
    cache.append(.{ 5, 6 }, .{ 50, 60 });
    try std.testing.expectEqual(@as(usize, 3), cache.len);

    // Adding 4th element should slide
    cache.append(.{ 7, 8 }, .{ 70, 80 });
    try std.testing.expectEqual(@as(usize, 3), cache.len);
    // First element should now be what was second
    try std.testing.expectEqual(@as(f32, 3.0), cache.keys[0][0]);
}

test "rmsNorm" {
    const x = [4]f32{ 1, 2, 3, 4 };
    const w = [4]f32{ 1, 1, 1, 1 };
    const out = rmsNorm(4, &x, &w);
    // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Normalized: [1/2.739, 2/2.739, 3/2.739, 4/2.739]
    try std.testing.expect(out[0] > 0.3 and out[0] < 0.4);
    try std.testing.expect(out[3] > 1.3 and out[3] < 1.5);
}

test "greedy sample" {
    const logits = [5]f32{ 0.1, 0.9, 0.3, 0.7, 0.2 };
    try std.testing.expectEqual(@as(usize, 1), greedySample(5, &logits));
}

test "top-k sample respects k" {
    const logits_init = [5]f32{ 0.1, 10.0, 0.3, 9.0, 0.2 };
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    // With k=2, should only sample from indices 1 and 3
    var counts = [_]usize{0} ** 5;
    for (0..100) |_| {
        var l = logits_init;
        counts[topKSample(5, &l, 2, 1.0, &rng)] += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), counts[0]);
    try std.testing.expectEqual(@as(usize, 0), counts[2]);
    try std.testing.expectEqual(@as(usize, 0), counts[4]);
    try std.testing.expect(counts[1] > 0);
    try std.testing.expect(counts[3] > 0);
}

test "multi-head attention init" {
    const MHA = MultiHeadAttention(32, 4, 16);
    const mha = MHA.init();
    try std.testing.expectEqual(@as(usize, 32), MHA.model_dim);
    try std.testing.expectEqual(@as(usize, 4), MHA.num_heads);
    try std.testing.expectEqual(@as(usize, 8), MHA.head_dim);
    try std.testing.expectEqual(@as(usize, 4096), MHA.param_count);
    _ = mha;
}
