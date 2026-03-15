// Group-Query Attention with Sliding Window (Qwen-style).
//
// GQA reduces memory by sharing KV heads across multiple query heads.
// Instead of 32 KV heads matching 32 Q heads, use 8 KV heads shared
// by groups of 4 Q heads each. This gives 4× KV cache savings.
//
// Sliding window attention limits each token to attend only to the
// last W tokens, reducing both memory and compute for long sequences.
// Optionally, the first N layers use full attention (for global context)
// while deeper layers use sliding window (for local patterns).
//
// KV cache is sized for KV heads, not query heads — the core saving.
//
// Reference: Qwen2.5 Technical Report, GQA + Sliding Window Attention

const std = @import("std");
const simd = @import("simd.zig");
const tensor = @import("tensor.zig");
const attention = @import("attention.zig");

/// Group-Query Attention with optional sliding window.
///
/// ModelDim:     full model embedding dimension
/// NumQHeads:    number of query heads (e.g., 8)
/// NumKVHeads:   number of KV heads (e.g., 2) — must divide NumQHeads evenly
/// HeadDim:      per-head dimension
/// MaxSeqLen:    maximum sequence length
/// WindowSize:   sliding window size (0 = full attention)
pub fn GroupQueryAttention(
    comptime ModelDim: usize,
    comptime NumQHeads: usize,
    comptime NumKVHeads: usize,
    comptime HeadDim: usize,
    comptime MaxSeqLen: usize,
    comptime WindowSize: usize,
) type {
    const QDim = NumQHeads * HeadDim;
    const KVDim = NumKVHeads * HeadDim;
    const GroupSize = NumQHeads / NumKVHeads; // Q heads per KV head
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HeadDim)));

    // Effective attention window
    const EffectiveWindow = if (WindowSize == 0) MaxSeqLen else WindowSize;

    return struct {
        // Query projection: ModelDim → QDim (one per Q head)
        wq: tensor.Tensor(QDim, ModelDim),
        // Key projection: ModelDim → KVDim (one per KV head)
        wk: tensor.Tensor(KVDim, ModelDim),
        // Value projection: ModelDim → KVDim
        wv: tensor.Tensor(KVDim, ModelDim),
        // Output projection: QDim → ModelDim
        wo: tensor.Tensor(ModelDim, QDim),

        const Self = @This();
        pub const model_dim = ModelDim;
        pub const num_q_heads = NumQHeads;
        pub const num_kv_heads = NumKVHeads;
        pub const head_dim = HeadDim;
        pub const group_size = GroupSize;
        pub const max_seq_len = MaxSeqLen;
        pub const window_size = WindowSize;
        pub const effective_window = EffectiveWindow;
        pub const q_dim = QDim;
        pub const kv_dim = KVDim;

        /// Memory savings from GQA: ratio of standard MHA cache to GQA cache.
        pub const kv_memory_ratio = @as(f32, @floatFromInt(NumQHeads)) / @as(f32, @floatFromInt(NumKVHeads));

        pub const param_count = QDim * ModelDim + // wq
            KVDim * ModelDim + // wk
            KVDim * ModelDim + // wv
            ModelDim * QDim; // wo

        pub fn init() Self {
            return .{
                .wq = tensor.Tensor(QDim, ModelDim).zeros(),
                .wk = tensor.Tensor(KVDim, ModelDim).zeros(),
                .wv = tensor.Tensor(KVDim, ModelDim).zeros(),
                .wo = tensor.Tensor(ModelDim, QDim).zeros(),
            };
        }

        /// Forward pass for a single token.
        ///
        /// x:         input embedding [ModelDim]
        /// kv_keys:   cached key vectors [len][KVDim]
        /// kv_values: cached value vectors [len][KVDim]
        /// pos:       current position in the sequence
        /// use_window: whether to apply sliding window (false = full attention)
        pub fn forward(
            self: *const Self,
            x: *const [ModelDim]f32,
            kv_keys: []const [KVDim]f32,
            kv_values: []const [KVDim]f32,
            pos: usize,
            use_window: bool,
        ) [ModelDim]f32 {
            // Project to Q, K, V
            const q_full = tensor.matvec(QDim, ModelDim, &self.wq, x);
            _ = tensor.matvec(KVDim, ModelDim, &self.wk, x); // current K (caller caches)
            _ = tensor.matvec(KVDim, ModelDim, &self.wv, x); // current V (caller caches)

            // Determine attention window
            const total_len = @min(pos + 1, kv_keys.len + 1);
            const window_start = if (use_window and WindowSize > 0 and total_len > EffectiveWindow)
                total_len - EffectiveWindow
            else
                0;
            const attend_len = total_len - window_start;

            // Per-head attention
            var head_outputs: [NumQHeads][HeadDim]f32 = undefined;

            for (0..NumQHeads) |qh| {
                const q_offset = qh * HeadDim;
                const kv_head = qh / GroupSize; // Which KV head this Q head uses
                const kv_offset = kv_head * HeadDim;

                // Extract this Q head's query
                var q_head: [HeadDim]f32 = undefined;
                for (0..HeadDim) |d| q_head[d] = q_full[q_offset + d];

                // Compute attention scores within window
                var scores: [MaxSeqLen]f32 = undefined;
                var v_heads: [MaxSeqLen][HeadDim]f32 = undefined;

                for (0..attend_len) |wi| {
                    const p = window_start + wi;

                    // Extract K head from cache
                    var k_head: [HeadDim]f32 = undefined;
                    var v_head: [HeadDim]f32 = undefined;

                    if (p < kv_keys.len) {
                        for (0..HeadDim) |d| {
                            k_head[d] = kv_keys[p][kv_offset + d];
                            v_head[d] = kv_values[p][kv_offset + d];
                        }
                    } else {
                        // Current position — use projected values
                        const k_cur = tensor.matvec(KVDim, ModelDim, &self.wk, x);
                        const v_cur = tensor.matvec(KVDim, ModelDim, &self.wv, x);
                        for (0..HeadDim) |d| {
                            k_head[d] = k_cur[kv_offset + d];
                            v_head[d] = v_cur[kv_offset + d];
                        }
                    }

                    scores[wi] = simd.vdot(HeadDim, &q_head, &k_head) * scale;
                    v_heads[wi] = v_head;
                }

                // Softmax
                var max_score: f32 = scores[0];
                for (1..attend_len) |i| {
                    if (scores[i] > max_score) max_score = scores[i];
                }
                var sum_exp: f32 = 0;
                for (0..attend_len) |i| {
                    scores[i] = @exp(scores[i] - max_score);
                    sum_exp += scores[i];
                }
                const inv = 1.0 / sum_exp;
                for (0..attend_len) |i| scores[i] *= inv;

                // Weighted sum
                var out: [HeadDim]f32 = [_]f32{0} ** HeadDim;
                for (0..attend_len) |i| {
                    for (0..HeadDim) |d| {
                        out[d] += scores[i] * v_heads[i][d];
                    }
                }
                head_outputs[qh] = out;
            }

            // Concatenate and project
            var concat: [QDim]f32 = undefined;
            for (0..NumQHeads) |h| {
                for (0..HeadDim) |d| {
                    concat[h * HeadDim + d] = head_outputs[h][d];
                }
            }

            return tensor.matvec(ModelDim, QDim, &self.wo, &concat);
        }
    };
}

/// GQA KV cache — sized for KV heads, not query heads.
/// This is the core memory saving of GQA.
pub fn GQAKVCache(comptime KVDim: usize, comptime MaxSeqLen: usize) type {
    return struct {
        keys: [MaxSeqLen][KVDim]f32,
        values: [MaxSeqLen][KVDim]f32,
        len: usize,

        const Self = @This();
        pub const kv_dim = KVDim;
        pub const max_seq_len = MaxSeqLen;

        pub fn init() Self {
            return .{
                .keys = [_][KVDim]f32{[_]f32{0} ** KVDim} ** MaxSeqLen,
                .values = [_][KVDim]f32{[_]f32{0} ** KVDim} ** MaxSeqLen,
                .len = 0,
            };
        }

        pub fn append(self: *Self, key: [KVDim]f32, value: [KVDim]f32) void {
            if (self.len >= MaxSeqLen) {
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

        /// Memory usage in bytes.
        pub fn memorySizeBytes() usize {
            return 2 * MaxSeqLen * KVDim * @sizeOf(f32);
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "GQA dimensions and group size" {
    // 8 Q heads, 2 KV heads → group size 4
    const GQA = GroupQueryAttention(64, 8, 2, 8, 32, 0);
    try std.testing.expectEqual(@as(usize, 4), GQA.group_size);
    try std.testing.expectEqual(@as(usize, 64), GQA.q_dim); // 8 * 8
    try std.testing.expectEqual(@as(usize, 16), GQA.kv_dim); // 2 * 8
    try std.testing.expectEqual(@as(f32, 4.0), GQA.kv_memory_ratio);
}

test "GQA KV cache memory savings" {
    // Standard MHA: 8 heads * 16 head_dim = 128 KV dim
    // GQA:          2 KV heads * 16 head_dim = 32 KV dim
    // 4× memory savings
    const q_heads = 8;
    const kv_heads = 2;
    const head_dim = 16;
    const max_seq = 1024;

    const mha_cache_bytes = 2 * max_seq * q_heads * head_dim * @sizeOf(f32);
    const gqa_cache_bytes = GQAKVCache(kv_heads * head_dim, max_seq).memorySizeBytes();

    const ratio = @as(f32, @floatFromInt(mha_cache_bytes)) / @as(f32, @floatFromInt(gqa_cache_bytes));
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), ratio, 0.01);
}

test "GQA forward produces correct output dims" {
    const GQA = GroupQueryAttention(16, 4, 2, 4, 8, 0);
    const gqa = GQA.init();

    const x = [_]f32{0.1} ** 16;
    var cache = GQAKVCache(8, 8).init(); // KVDim = 2 * 4 = 8

    // Forward with empty cache (first position, full attention)
    const out = gqa.forward(&x, cache.keys[0..cache.len], cache.values[0..cache.len], 0, false);
    try std.testing.expectEqual(@as(usize, 16), out.len);
}

test "GQA sliding window limits attention" {
    // Window size 4: only attend to last 4 tokens
    const GQA = GroupQueryAttention(16, 4, 2, 4, 16, 4);
    try std.testing.expectEqual(@as(usize, 4), GQA.window_size);
    try std.testing.expectEqual(@as(usize, 4), GQA.effective_window);

    // Full attention: window_size = 0
    const GQAFull = GroupQueryAttention(16, 4, 2, 4, 16, 0);
    try std.testing.expectEqual(@as(usize, 0), GQAFull.window_size);
    try std.testing.expectEqual(@as(usize, 16), GQAFull.effective_window);
}

test "GQA forward with sliding window" {
    const GQA = GroupQueryAttention(8, 2, 1, 4, 8, 3);
    var gqa = GQA.init();

    // Set some weights for non-trivial behavior
    gqa.wq.data[0][0] = 1.0;
    gqa.wk.data[0][0] = 1.0;
    gqa.wv.data[0][0] = 1.0;

    var cache = GQAKVCache(4, 8).init();

    // Fill cache with 5 positions
    for (0..5) |i| {
        var k: [4]f32 = [_]f32{0} ** 4;
        k[0] = @as(f32, @floatFromInt(i)) * 0.1;
        cache.append(k, k);
    }

    const x = [_]f32{0.1} ** 8;
    // With window=3, should only attend to last 3 of cache
    const out = gqa.forward(&x, cache.keys[0..cache.len], cache.values[0..cache.len], 5, true);
    for (out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "GQAKVCache append and eviction" {
    var cache = GQAKVCache(4, 3).init();
    try std.testing.expectEqual(@as(usize, 0), cache.len);

    cache.append([_]f32{ 1, 2, 3, 4 }, [_]f32{ 10, 20, 30, 40 });
    cache.append([_]f32{ 5, 6, 7, 8 }, [_]f32{ 50, 60, 70, 80 });
    cache.append([_]f32{ 9, 10, 11, 12 }, [_]f32{ 90, 100, 110, 120 });
    try std.testing.expectEqual(@as(usize, 3), cache.len);

    // Eviction on overflow
    cache.append([_]f32{ 13, 14, 15, 16 }, [_]f32{ 130, 140, 150, 160 });
    try std.testing.expectEqual(@as(usize, 3), cache.len);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cache.keys[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), cache.keys[2][0], 1e-5);
}

test "GQA param count less than full MHA" {
    // GQA with 8 Q heads, 2 KV heads
    const GQA = GroupQueryAttention(64, 8, 2, 8, 32, 0);
    // MHA with 8 Q heads, 8 KV heads (standard)
    const MHA = GroupQueryAttention(64, 8, 8, 8, 32, 0);

    // GQA should have fewer params due to smaller K and V projections
    try std.testing.expect(GQA.param_count < MHA.param_count);
}
