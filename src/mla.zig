// Multi-Head Latent Attention (DeepSeek-V3 style).
//
// The key insight: compress the KV cache from [seq_len × n_heads × head_dim]
// to [seq_len × latent_dim]. This trades 4× more compute for 8-28× less
// memory bandwidth — a massive win on memory-bound inference.
//
// Architecture:
//   DownProject: hidden_dim → latent_dim  (compress KV into low-rank latent)
//   UpProject:   latent_dim → K, V        (expand back for attention computation)
//
// The latent cache stores only the compressed representation. During attention,
// we expand back to full K,V on the fly. This is compute-bound (cheap) rather
// than memory-bound (expensive).
//
// Reference: DeepSeek-V3 Technical Report, Section 3.1

const std = @import("std");
const simd = @import("simd.zig");
const tensor = @import("tensor.zig");
const attention = @import("attention.zig");

/// Multi-Head Latent Attention with comptime-known dimensions.
///
/// ModelDim:   full model embedding dimension
/// NumHeads:   number of attention heads
/// HeadDim:    per-head dimension (ModelDim / NumHeads)
/// LatentDim:  compressed KV dimension (typically HeadDim / 2 to HeadDim)
/// MaxSeqLen:  maximum sequence length
pub fn MultiHeadLatentAttention(
    comptime ModelDim: usize,
    comptime NumHeads: usize,
    comptime HeadDim: usize,
    comptime LatentDim: usize,
    comptime MaxSeqLen: usize,
) type {
    // KV dimension = NumHeads * HeadDim
    const KVDim = NumHeads * HeadDim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HeadDim)));

    return struct {
        // Query projection: ModelDim → KVDim
        wq: tensor.Tensor(KVDim, ModelDim),

        // Down-projection: ModelDim → LatentDim (compress for cache)
        w_down: tensor.Tensor(LatentDim, ModelDim),

        // Up-projections: LatentDim → KVDim (expand K and V from latent)
        w_up_k: tensor.Tensor(KVDim, LatentDim),
        w_up_v: tensor.Tensor(KVDim, LatentDim),

        // Output projection: KVDim → ModelDim
        wo: tensor.Tensor(ModelDim, KVDim),

        const Self = @This();
        pub const model_dim = ModelDim;
        pub const num_heads = NumHeads;
        pub const head_dim = HeadDim;
        pub const latent_dim = LatentDim;
        pub const max_seq_len = MaxSeqLen;
        pub const kv_dim = KVDim;

        /// Standard MHA KV cache size (per position, in f32 elements).
        pub const standard_kv_size = 2 * KVDim;
        /// MLA latent cache size (per position, in f32 elements).
        pub const latent_kv_size = LatentDim;
        /// Compression ratio: how much smaller the MLA cache is.
        pub const compression_ratio = @as(f32, @floatFromInt(standard_kv_size)) / @as(f32, @floatFromInt(latent_kv_size));

        pub const param_count = KVDim * ModelDim + // wq
            LatentDim * ModelDim + // w_down
            KVDim * LatentDim + // w_up_k
            KVDim * LatentDim + // w_up_v
            ModelDim * KVDim; // wo

        pub fn init() Self {
            return .{
                .wq = tensor.Tensor(KVDim, ModelDim).zeros(),
                .w_down = tensor.Tensor(LatentDim, ModelDim).zeros(),
                .w_up_k = tensor.Tensor(KVDim, LatentDim).zeros(),
                .w_up_v = tensor.Tensor(KVDim, LatentDim).zeros(),
                .wo = tensor.Tensor(ModelDim, KVDim).zeros(),
            };
        }

        /// Compress input to latent representation for KV cache storage.
        pub fn compress(self: *const Self, x: *const [ModelDim]f32) [LatentDim]f32 {
            return tensor.matvec(LatentDim, ModelDim, &self.w_down, x);
        }

        /// Expand latent to full key vector.
        pub fn expandKey(self: *const Self, latent: *const [LatentDim]f32) [KVDim]f32 {
            return tensor.matvec(KVDim, LatentDim, &self.w_up_k, latent);
        }

        /// Expand latent to full value vector.
        pub fn expandValue(self: *const Self, latent: *const [LatentDim]f32) [KVDim]f32 {
            return tensor.matvec(KVDim, LatentDim, &self.w_up_v, latent);
        }

        /// Forward pass for a single token position.
        ///
        /// x: input embedding [ModelDim]
        /// latent_cache: compressed KV cache [len][LatentDim]
        /// pos: current position in the sequence
        pub fn forward(
            self: *const Self,
            x: *const [ModelDim]f32,
            latent_cache: []const [LatentDim]f32,
            pos: usize,
        ) [ModelDim]f32 {
            // Project input to query
            const q_full = tensor.matvec(KVDim, ModelDim, &self.wq, x);

            // Compress current input to latent for cache
            const cur_latent = self.compress(x);

            // Per-head attention computation
            var head_outputs: [NumHeads][HeadDim]f32 = undefined;
            const seq_len = @min(pos + 1, latent_cache.len + 1);

            for (0..NumHeads) |h| {
                const offset = h * HeadDim;

                // Extract this head's query
                var q_head: [HeadDim]f32 = undefined;
                for (0..HeadDim) |d| q_head[d] = q_full[offset + d];

                // Compute attention scores against all cached positions
                var scores: [MaxSeqLen]f32 = undefined;
                var values_expanded: [MaxSeqLen][HeadDim]f32 = undefined;

                for (0..seq_len) |p| {
                    // Get latent for this position
                    var lat: [LatentDim]f32 = undefined;
                    if (p < latent_cache.len) {
                        lat = latent_cache[p];
                    } else {
                        lat = cur_latent;
                    }

                    // Expand to K and V on the fly
                    const k_full = self.expandKey(&lat);
                    const v_full = self.expandValue(&lat);

                    // Extract this head's K and V
                    var k_head: [HeadDim]f32 = undefined;
                    var v_head: [HeadDim]f32 = undefined;
                    for (0..HeadDim) |d| {
                        k_head[d] = k_full[offset + d];
                        v_head[d] = v_full[offset + d];
                    }

                    scores[p] = simd.vdot(HeadDim, &q_head, &k_head) * scale;
                    values_expanded[p] = v_head;
                }

                // Softmax over scores
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
                for (0..seq_len) |i| scores[i] *= inv_sum;

                // Weighted sum of values
                var out: [HeadDim]f32 = [_]f32{0} ** HeadDim;
                for (0..seq_len) |i| {
                    for (0..HeadDim) |d| {
                        out[d] += scores[i] * values_expanded[i][d];
                    }
                }
                head_outputs[h] = out;
            }

            // Concatenate heads
            var concat: [KVDim]f32 = undefined;
            for (0..NumHeads) |h| {
                for (0..HeadDim) |d| {
                    concat[h * HeadDim + d] = head_outputs[h][d];
                }
            }

            return tensor.matvec(ModelDim, KVDim, &self.wo, &concat);
        }
    };
}

/// Latent KV cache — stores only the compressed latent vectors.
/// This is the key memory saving: LatentDim << NumHeads * HeadDim.
pub fn LatentKVCache(comptime LatentDim: usize, comptime MaxSeqLen: usize) type {
    return struct {
        latents: [MaxSeqLen][LatentDim]f32,
        len: usize,

        const Self = @This();

        pub fn init() Self {
            return .{
                .latents = [_][LatentDim]f32{[_]f32{0} ** LatentDim} ** MaxSeqLen,
                .len = 0,
            };
        }

        pub fn append(self: *Self, latent: [LatentDim]f32) void {
            if (self.len >= MaxSeqLen) {
                // Sliding window eviction
                for (0..MaxSeqLen - 1) |i| {
                    self.latents[i] = self.latents[i + 1];
                }
                self.latents[MaxSeqLen - 1] = latent;
            } else {
                self.latents[self.len] = latent;
                self.len += 1;
            }
        }

        pub fn reset(self: *Self) void {
            self.len = 0;
        }

        /// Memory usage in bytes for this latent cache.
        pub fn memorySizeBytes() usize {
            return MaxSeqLen * LatentDim * @sizeOf(f32);
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "MLA compression ratio" {
    // 8 heads, 16 head_dim, latent_dim=8
    // Standard KV: 2 * 8 * 16 = 256 floats per position
    // Latent KV: 8 floats per position
    // Compression: 32×
    const MLA = MultiHeadLatentAttention(128, 8, 16, 8, 32);
    try std.testing.expectEqual(@as(usize, 256), MLA.standard_kv_size);
    try std.testing.expectEqual(@as(usize, 8), MLA.latent_kv_size);
    try std.testing.expect(MLA.compression_ratio >= 30.0);
}

test "MLA compress and expand roundtrip structure" {
    // Test that compress produces LatentDim output and expand produces KVDim output
    const MLA = MultiHeadLatentAttention(16, 2, 8, 4, 8);
    var mla = MLA.init();

    // Set w_down to identity-like mapping (first 4 dims)
    for (0..4) |i| {
        mla.w_down.data[i][i] = 1.0;
    }
    // Set w_up_k to expand back
    for (0..4) |i| {
        mla.w_up_k.data[i][i] = 1.0;
    }

    var x: [16]f32 = [_]f32{0} ** 16;
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    const latent = mla.compress(&x);
    try std.testing.expectEqual(@as(usize, 4), latent.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), latent[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), latent[1], 1e-5);

    const k_expanded = mla.expandKey(&latent);
    try std.testing.expectEqual(@as(usize, 16), k_expanded.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), k_expanded[0], 1e-5);
}

test "MLA forward produces correct output dimensions" {
    const MLA = MultiHeadLatentAttention(16, 2, 8, 4, 8);
    const mla = MLA.init();

    var x: [16]f32 = [_]f32{0.1} ** 16;
    var cache = LatentKVCache(4, 8).init();

    // Forward with empty cache (first position)
    const out = mla.forward(&x, cache.latents[0..cache.len], 0);
    try std.testing.expectEqual(@as(usize, 16), out.len);
}

test "MLA forward with cached positions" {
    const MLA = MultiHeadLatentAttention(16, 2, 8, 4, 8);
    var mla = MLA.init();

    // Set some non-zero weights so attention does something
    for (0..4) |i| {
        mla.w_down.data[i][i] = 0.5;
        mla.w_up_k.data[i][i] = 1.0;
        mla.w_up_v.data[i][i] = 1.0;
    }
    mla.wq.data[0][0] = 1.0;

    var cache = LatentKVCache(4, 8).init();

    // Add some latent vectors to cache
    cache.append([_]f32{ 0.1, 0.2, 0.3, 0.4 });
    cache.append([_]f32{ 0.5, 0.6, 0.7, 0.8 });

    var x: [16]f32 = [_]f32{0.1} ** 16;
    const out = mla.forward(&x, cache.latents[0..cache.len], 2);

    // Output should be finite and from ModelDim dimensions
    for (out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "LatentKVCache append and sliding window" {
    var cache = LatentKVCache(4, 3).init();
    try std.testing.expectEqual(@as(usize, 0), cache.len);

    cache.append([_]f32{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 1), cache.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cache.latents[0][0], 1e-5);

    cache.append([_]f32{ 5, 6, 7, 8 });
    cache.append([_]f32{ 9, 10, 11, 12 });
    try std.testing.expectEqual(@as(usize, 3), cache.len);

    // Sliding window: 4th append evicts oldest
    cache.append([_]f32{ 13, 14, 15, 16 });
    try std.testing.expectEqual(@as(usize, 3), cache.len);
    // First element should now be what was second
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cache.latents[0][0], 1e-5);
    // Last element should be the new one
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), cache.latents[2][0], 1e-5);
}

test "MLA memory savings benchmark" {
    // Realistic config: 32 heads, 128 head_dim, latent_dim=512, seq_len=4096
    // Standard: 4096 * 2 * 32 * 128 * 4 bytes = 128 MB
    // MLA:      4096 * 512 * 4 bytes = 8 MB
    // 16× savings

    const seq_len = 4096;
    const num_heads = 32;
    const head_dim = 128;
    const latent_dim = 512;

    const standard_bytes = seq_len * 2 * num_heads * head_dim * @sizeOf(f32);
    const mla_bytes = seq_len * latent_dim * @sizeOf(f32);

    const ratio = @as(f32, @floatFromInt(standard_bytes)) / @as(f32, @floatFromInt(mla_bytes));
    try std.testing.expect(ratio >= 16.0);

    // Also verify with the cache type
    const latent_cache_size = LatentKVCache(latent_dim, seq_len).memorySizeBytes();
    try std.testing.expectEqual(mla_bytes, latent_cache_size);
}

test "MLA param count" {
    const MLA = MultiHeadLatentAttention(64, 4, 16, 8, 16);
    // wq: 64*64=4096, w_down: 8*64=512, w_up_k: 64*8=512, w_up_v: 64*8=512, wo: 64*64=4096
    try std.testing.expectEqual(@as(usize, 4096 + 512 + 512 + 512 + 4096), MLA.param_count);
}
