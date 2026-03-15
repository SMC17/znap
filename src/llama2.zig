// Llama2 model loader and inference — compatible with Karpathy's llama2.c format.
//
// Loads pre-trained models (stories15M, stories42M, stories110M) and
// generates text using the znap inference engine.
//
// Binary format (from llama2.c):
//   Config: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len (7 × i32)
//   Weights: token_embedding, rms_att_weight, wq, wk, wv, wo,
//            rms_ffn_weight, w1, w2, w3, rms_final_weight, freq_cis_real, freq_cis_imag, wcls

const std = @import("std");
const simd = @import("simd.zig");
const gguf_mod = @import("gguf.zig");
const int8 = @import("int8_matmul.zig");
const w8a8 = @import("w8a8.zig");
const neon = @import("neon.zig");

pub const Config = struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: i32, // negative = shared weights with token embedding
    seq_len: u32,
    rope_freq_base: f32 = 10000.0,

    pub fn headDim(self: Config) u32 {
        return self.dim / self.n_heads;
    }

    pub fn kvDim(self: Config) u32 {
        return (self.dim * self.n_kv_heads) / self.n_heads;
    }

    pub fn kvMul(self: Config) u32 {
        return self.n_heads / self.n_kv_heads;
    }
};

pub const Weights = struct {
    // All weights stored as flat f32 slices into the mmap'd file
    token_embedding: []const f32,
    rms_att_weight: []const f32, // [n_layers * dim]
    wq: []const f32, // [n_layers * dim * dim]
    wk: []const f32, // [n_layers * dim * kv_dim]
    wv: []const f32, // [n_layers * dim * kv_dim]
    wo: []const f32, // [n_layers * dim * dim]
    rms_ffn_weight: []const f32, // [n_layers * dim]
    w1: []const f32, // [n_layers * hidden_dim * dim]
    w2: []const f32, // [n_layers * dim * hidden_dim]
    w3: []const f32, // [n_layers * hidden_dim * dim]
    rms_final_weight: []const f32, // [dim]
    wcls: []const f32, // [vocab_size * dim] or shared with token_embedding
};

pub const RunState = struct {
    x: []f32, // [dim] activation at current position
    xb: []f32, // [dim] buffer
    xb2: []f32, // [dim] buffer 2
    hb: []f32, // [hidden_dim] hidden buffer for ffn
    hb2: []f32, // [hidden_dim] hidden buffer 2
    q: []f32, // [dim] query
    k: []f32, // [kv_dim] key
    v: []f32, // [kv_dim] value
    att: []f32, // [n_heads * seq_len] attention scores
    logits: []f32, // [vocab_size] output logits
    key_cache: []f32, // [n_layers * seq_len * kv_dim]
    value_cache: []f32, // [n_layers * seq_len * kv_dim]

    pub fn init(config: Config, allocator: std.mem.Allocator) !RunState {
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const kv_dim = config.kvDim();
        const vocab = @as(u32, @intCast(if (config.vocab_size < 0) -config.vocab_size else config.vocab_size));
        const n_layers = config.n_layers;
        const seq_len = config.seq_len;
        const n_heads = config.n_heads;

        return .{
            .x = try allocator.alloc(f32, dim),
            .xb = try allocator.alloc(f32, dim),
            .xb2 = try allocator.alloc(f32, dim),
            .hb = try allocator.alloc(f32, hidden_dim),
            .hb2 = try allocator.alloc(f32, hidden_dim),
            .q = try allocator.alloc(f32, dim),
            .k = try allocator.alloc(f32, kv_dim),
            .v = try allocator.alloc(f32, kv_dim),
            .att = try allocator.alloc(f32, n_heads * seq_len),
            .logits = try allocator.alloc(f32, vocab),
            .key_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim),
            .value_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim),
        };
    }

    pub fn deinit(self: *RunState, allocator: std.mem.Allocator) void {
        allocator.free(self.x);
        allocator.free(self.xb);
        allocator.free(self.xb2);
        allocator.free(self.hb);
        allocator.free(self.hb2);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.att);
        allocator.free(self.logits);
        allocator.free(self.key_cache);
        allocator.free(self.value_cache);
    }
};

// ── RMS Normalization ────────────────────────────────────────────

fn rmsNorm(out: []f32, x: []const f32, weight: []const f32) void {
    const n = x.len;
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    ss = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + 1e-5);
    for (0..n) |i| out[i] = weight[i] * (ss * x[i]);
}

// ── Softmax ──────────────────────────────────────────────────────

fn softmax(x: []f32) void {
    var max_val: f32 = x[0];
    for (x[1..]) |v| if (v > max_val) { max_val = v; };
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

// ── Matrix-Vector Multiply (SIMD-accelerated) ────────────────────

/// SIMD dot product of a single row — 8-way unrolled for maximum ILP.
fn dotRow(row: []const f32, x: []const f32, n: usize) f32 {
    const V = 4;
    // 8 accumulators to maximize instruction-level parallelism
    // M1 can issue 4 FMLA per cycle with enough independent accumulators
    var a0: @Vector(V, f32) = @splat(0);
    var a1: @Vector(V, f32) = @splat(0);
    var a2: @Vector(V, f32) = @splat(0);
    var a3: @Vector(V, f32) = @splat(0);
    var a4: @Vector(V, f32) = @splat(0);
    var a5: @Vector(V, f32) = @splat(0);
    var a6: @Vector(V, f32) = @splat(0);
    var a7: @Vector(V, f32) = @splat(0);
    var j: usize = 0;

    // Process 32 floats per iteration (8 × 4-wide FMLA)
    while (j + 32 <= n) : (j += 32) {
        a0 += @as(@Vector(V, f32), row[j..][0..4].*) * @as(@Vector(V, f32), x[j..][0..4].*);
        a1 += @as(@Vector(V, f32), row[j + 4 ..][0..4].*) * @as(@Vector(V, f32), x[j + 4 ..][0..4].*);
        a2 += @as(@Vector(V, f32), row[j + 8 ..][0..4].*) * @as(@Vector(V, f32), x[j + 8 ..][0..4].*);
        a3 += @as(@Vector(V, f32), row[j + 12 ..][0..4].*) * @as(@Vector(V, f32), x[j + 12 ..][0..4].*);
        a4 += @as(@Vector(V, f32), row[j + 16 ..][0..4].*) * @as(@Vector(V, f32), x[j + 16 ..][0..4].*);
        a5 += @as(@Vector(V, f32), row[j + 20 ..][0..4].*) * @as(@Vector(V, f32), x[j + 20 ..][0..4].*);
        a6 += @as(@Vector(V, f32), row[j + 24 ..][0..4].*) * @as(@Vector(V, f32), x[j + 24 ..][0..4].*);
        a7 += @as(@Vector(V, f32), row[j + 28 ..][0..4].*) * @as(@Vector(V, f32), x[j + 28 ..][0..4].*);
    }

    // Reduce: merge 8 accumulators → 1
    var sum = @reduce(.Add, (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7));

    // Handle remaining 4-at-a-time
    while (j + V <= n) : (j += V) {
        sum += @reduce(.Add, @as(@Vector(V, f32), row[j..][0..V].*) * @as(@Vector(V, f32), x[j..][0..V].*));
    }

    while (j < n) : (j += 1) sum += row[j] * x[j];

    return sum;
}

/// Multithreaded worker for parallel matmul rows.
const MatmulWork = struct {
    out: []f32,
    w: []const f32,
    x: []const f32,
    n: usize,
    start_row: usize,
    end_row: usize,
};

fn matmulWorker(work: MatmulWork) void {
    for (work.start_row..work.end_row) |i| {
        work.out[i] = dotRow(work.w[i * work.n ..][0..work.n], work.x, work.n);
    }
}

fn matmul(out: []f32, x: []const f32, w: []const f32, n: usize) void {
    const d = out.len;

    // Only multithread for truly large matmuls (vocab projection).
    // Thread spawn overhead (~20μs) must be amortized over enough work.
    // Vocab projections: 32K-49K output rows × 288-576 cols = 9M-28M elements.
    // Layer matmuls: 288-1536 rows × 288-576 cols = 83K-885K elements.
    // Threshold: 2M elements — only catches vocab projections.
    if (d * n < 2_000_000) {
        for (0..d) |i| {
            out[i] = dotRow(w[i * n ..][0..n], x, n);
        }
        return;
    }

    // Multithreaded: split rows across available cores
    const n_threads = @min(std.Thread.getCpuCount() catch 1, 8);
    if (n_threads <= 1) {
        for (0..d) |i| {
            out[i] = dotRow(w[i * n ..][0..n], x, n);
        }
        return;
    }

    const rows_per_thread = d / n_threads;
    var threads: [8]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..n_threads) |t| {
        const start = t * rows_per_thread;
        const end = if (t == n_threads - 1) d else (t + 1) * rows_per_thread;
        if (start >= end) continue;

        threads[t] = std.Thread.spawn(.{}, matmulWorker, .{MatmulWork{
            .out = out,
            .w = w,
            .x = x,
            .n = n,
            .start_row = start,
            .end_row = end,
        }}) catch {
            // Fallback: do this chunk on current thread
            matmulWorker(.{ .out = out, .w = w, .x = x, .n = n, .start_row = start, .end_row = end });
            continue;
        };
        spawned += 1;
    }

    for (0..spawned) |t| threads[t].join();
}

/// Quantized weight storage — raw bytes + metadata for native quantized matmul.
pub const QuantizedTensor = struct {
    raw: []const u8,
    tensor_type: u32,
    n_cols: usize,
};

pub const QuantizedLayer = struct {
    wq: QuantizedTensor,
    wk: QuantizedTensor,
    wv: QuantizedTensor,
    wo: QuantizedTensor,
    w1: QuantizedTensor,
    w2: QuantizedTensor,
    w3: QuantizedTensor,
};

pub const QuantizedWeights = struct {
    layers: []QuantizedLayer,
    raw_wcls: ?[]const u8,
    wcls_type: u32,
    wcls_cols: usize,
    wcls_rows: usize,
};

const quant_matmul = @import("quant_matmul.zig");

/// Matrix-vector multiply using raw quantized data (no dequantization).
fn matmulRaw(out: []f32, x: []const f32, raw_data: []const u8, n_cols: usize, tensor_type: u32) void {
    const n_rows = out.len;
    const rb = quant_matmul.rowBytes(n_cols, tensor_type);

    for (0..n_rows) |i| {
        out[i] = quant_matmul.dotQuantizedRow(raw_data[i * rb ..][0..rb], x[0..n_cols], n_cols, tensor_type);
    }
}

// ── Transformer Forward Pass ─────────────────────────────────────

pub fn forward(config: Config, weights: *const Weights, state: *RunState, token: u32, pos: u32) void {
    const dim = config.dim;
    const hidden_dim = config.hidden_dim;
    const head_dim = config.headDim();
    const kv_dim = config.kvDim();
    const kv_mul = config.kvMul();
    const n_heads = config.n_heads;
    const n_layers = config.n_layers;
    const seq_len = config.seq_len;

    // Token embedding
    const emb = weights.token_embedding[token * dim .. (token + 1) * dim];
    @memcpy(state.x, emb);

    // Forward through all layers
    for (0..n_layers) |l| {
        // Attention RMSNorm
        rmsNorm(state.xb, state.x, weights.rms_att_weight[l * dim .. (l + 1) * dim]);

        // QKV projections
        matmul(state.q, state.xb, weights.wq[l * dim * dim ..], dim);
        matmul(state.k, state.xb, weights.wk[l * dim * kv_dim ..], dim);
        matmul(state.v, state.xb, weights.wv[l * dim * kv_dim ..], dim);

        // RoPE
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const q0 = q_head[i];
                const q1 = q_head[i + 1];
                q_head[i] = q0 * cos_a - q1 * sin_a;
                q_head[i + 1] = q0 * sin_a + q1 * cos_a;
            }
        }
        // Apply RoPE to K
        for (0..config.n_kv_heads) |h| {
            const k_head = state.k[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const k0 = k_head[i];
                const k1 = k_head[i + 1];
                k_head[i] = k0 * cos_a - k1 * sin_a;
                k_head[i + 1] = k0 * sin_a + k1 * cos_a;
            }
        }

        // Cache K, V
        const cache_offset = l * seq_len * kv_dim + pos * kv_dim;
        @memcpy(state.key_cache[cache_offset .. cache_offset + kv_dim], state.k);
        @memcpy(state.value_cache[cache_offset .. cache_offset + kv_dim], state.v);

        // Multi-head attention
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            const att = state.att[h * seq_len .. h * seq_len + pos + 1];

            // Attention scores
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const k_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const k_vec = state.key_cache[k_offset .. k_offset + head_dim];
                var score: f32 = 0;
                for (0..head_dim) |d| score += q_head[d] * k_vec[d];
                att[t] = score / @sqrt(@as(f32, @floatFromInt(head_dim)));
            }

            softmax(att);

            // Weighted sum of values
            const xb_head = state.xb[h * head_dim .. (h + 1) * head_dim];
            @memset(xb_head, 0);
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const v_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const v_vec = state.value_cache[v_offset .. v_offset + head_dim];
                for (0..head_dim) |d| {
                    xb_head[d] += att[t] * v_vec[d];
                }
            }
        }

        // Output projection + residual
        matmul(state.xb2, state.xb, weights.wo[l * dim * dim ..], dim);
        for (0..dim) |i| state.x[i] += state.xb2[i];

        // FFN
        rmsNorm(state.xb, state.x, weights.rms_ffn_weight[l * dim .. (l + 1) * dim]);
        matmul(state.hb, state.xb, weights.w1[l * hidden_dim * dim ..], dim);
        matmul(state.hb2, state.xb, weights.w3[l * hidden_dim * dim ..], dim);

        // SwiGLU
        for (0..hidden_dim) |i| {
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + @exp(-state.hb[i]))) * state.hb2[i];
        }

        matmul(state.xb, state.hb, weights.w2[l * dim * hidden_dim ..], hidden_dim);
        for (0..dim) |i| state.x[i] += state.xb[i];
    }

    // Final RMSNorm
    rmsNorm(state.x, state.x, weights.rms_final_weight);

    // Classifier
    matmul(state.logits, state.x, weights.wcls, dim);
}

/// INT8 weight set — requantized from f32 for fast inference.
pub const Int8Weights = struct {
    wq: []u8, // [n_layers × dim × (4 + dim)]
    wk: []u8,
    wv: []u8,
    wo: []u8,
    w1: []u8,
    w2: []u8,
    w3: []u8,
    wcls: []u8,

    /// Requantize all weight matrices from f32 Weights to INT8.
    pub fn fromWeights(w: *const Weights, config: Config, allocator: std.mem.Allocator) !Int8Weights {
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const kv_dim = config.kvDim();
        const n_layers = config.n_layers;
        const vocab = @as(u32, @intCast(if (config.vocab_size < 0) -config.vocab_size else config.vocab_size));

        return .{
            .wq = try int8.requantizeToInt8(w.wq, n_layers * dim, dim, allocator),
            .wk = try int8.requantizeToInt8(w.wk, n_layers * kv_dim, dim, allocator),
            .wv = try int8.requantizeToInt8(w.wv, n_layers * kv_dim, dim, allocator),
            .wo = try int8.requantizeToInt8(w.wo, n_layers * dim, dim, allocator),
            .w1 = try int8.requantizeToInt8(w.w1, n_layers * hidden_dim, dim, allocator),
            .w2 = try int8.requantizeToInt8(w.w2, n_layers * dim, hidden_dim, allocator),
            .w3 = try int8.requantizeToInt8(w.w3, n_layers * hidden_dim, dim, allocator),
            .wcls = try int8.requantizeToInt8(w.wcls, vocab, dim, allocator),
        };
    }

    pub fn deinit(self: *Int8Weights, allocator: std.mem.Allocator) void {
        allocator.free(self.wq);
        allocator.free(self.wk);
        allocator.free(self.wv);
        allocator.free(self.wo);
        allocator.free(self.w1);
        allocator.free(self.w2);
        allocator.free(self.w3);
        allocator.free(self.wcls);
    }
};

/// INT8 matmul wrapper.
fn matmulI8(out: []f32, x: []const f32, i8_data: []const u8, n_cols: usize) void {
    const n_rows = out.len;
    int8.matmulInt8Threaded(i8_data, x, out, n_cols, n_rows);
}

/// Forward pass using INT8 quantized weights.
/// Embeddings and norms are still f32. Weight matmuls use INT8.
pub fn forwardInt8(config: Config, weights: *const Weights, i8w: *const Int8Weights, state: *RunState, token: u32, pos: u32) void {
    const dim = config.dim;
    const hidden_dim = config.hidden_dim;
    const head_dim = config.headDim();
    const kv_dim = config.kvDim();
    const kv_mul = config.kvMul();
    const n_heads = config.n_heads;
    const n_layers = config.n_layers;
    const seq_len = config.seq_len;

    const i8_row_q = int8.INT8_HEADER_SIZE + dim;
    const i8_row_kv = int8.INT8_HEADER_SIZE + dim; // K,V input is still dim
    const i8_row_o = int8.INT8_HEADER_SIZE + dim;
    const i8_row_w1 = int8.INT8_HEADER_SIZE + dim;
    const i8_row_w2 = int8.INT8_HEADER_SIZE + hidden_dim;
    const i8_row_w3 = int8.INT8_HEADER_SIZE + dim;

    // Token embedding (f32)
    const emb = weights.token_embedding[token * dim .. (token + 1) * dim];
    @memcpy(state.x, emb);

    for (0..n_layers) |l| {
        // Attention RMSNorm (f32)
        rmsNorm(state.xb, state.x, weights.rms_att_weight[l * dim .. (l + 1) * dim]);

        // QKV — INT8 matmul
        matmulI8(state.q, state.xb, i8w.wq[l * dim * i8_row_q ..], dim);
        matmulI8(state.k, state.xb, i8w.wk[l * kv_dim * i8_row_kv ..], dim);
        matmulI8(state.v, state.xb, i8w.wv[l * kv_dim * i8_row_kv ..], dim);

        // RoPE (same as f32 path)
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const q0 = q_head[i];
                const q1 = q_head[i + 1];
                q_head[i] = q0 * cos_a - q1 * sin_a;
                q_head[i + 1] = q0 * sin_a + q1 * cos_a;
            }
        }
        for (0..config.n_kv_heads) |h| {
            const k_head = state.k[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const k0 = k_head[i];
                const k1 = k_head[i + 1];
                k_head[i] = k0 * cos_a - k1 * sin_a;
                k_head[i + 1] = k0 * sin_a + k1 * cos_a;
            }
        }

        // KV cache
        const cache_offset = l * seq_len * kv_dim + pos * kv_dim;
        @memcpy(state.key_cache[cache_offset .. cache_offset + kv_dim], state.k);
        @memcpy(state.value_cache[cache_offset .. cache_offset + kv_dim], state.v);

        // Attention (f32 on cached K,V)
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            const att = state.att[h * seq_len .. h * seq_len + pos + 1];
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const k_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const k_vec = state.key_cache[k_offset .. k_offset + head_dim];
                var score: f32 = 0;
                for (0..head_dim) |d_idx| score += q_head[d_idx] * k_vec[d_idx];
                att[t] = score / @sqrt(@as(f32, @floatFromInt(head_dim)));
            }
            softmax(att);
            const xb_head = state.xb[h * head_dim .. (h + 1) * head_dim];
            @memset(xb_head, 0);
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const v_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const v_vec = state.value_cache[v_offset .. v_offset + head_dim];
                for (0..head_dim) |d_idx| xb_head[d_idx] += att[t] * v_vec[d_idx];
            }
        }

        // Output projection — INT8
        matmulI8(state.xb2, state.xb, i8w.wo[l * dim * i8_row_o ..], dim);
        for (0..dim) |i| state.x[i] += state.xb2[i];

        // FFN — INT8
        rmsNorm(state.xb, state.x, weights.rms_ffn_weight[l * dim .. (l + 1) * dim]);
        matmulI8(state.hb, state.xb, i8w.w1[l * hidden_dim * i8_row_w1 ..], dim);
        matmulI8(state.hb2, state.xb, i8w.w3[l * hidden_dim * i8_row_w3 ..], dim);

        // SwiGLU
        for (0..hidden_dim) |i| {
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + @exp(-state.hb[i]))) * state.hb2[i];
        }

        matmulI8(state.xb, state.hb, i8w.w2[l * dim * i8_row_w2 ..], hidden_dim);
        for (0..dim) |i| state.x[i] += state.xb[i];
    }

    // Final RMSNorm (f32)
    rmsNorm(state.x, state.x, weights.rms_final_weight);

    // Classifier — INT8
    matmulI8(state.logits, state.x, i8w.wcls, dim);
}

/// Forward pass using quantized weights directly (no dequantization).
/// Uses f32 for embeddings and norms, raw quantized data for matmuls.
pub fn forwardQuantized(
    config: Config,
    weights: *const Weights, // For embeddings and norms (always f32)
    qw: *const QuantizedWeights,
    state: *RunState,
    token: u32,
    pos: u32,
) void {
    const dim = config.dim;
    const hidden_dim = config.hidden_dim;
    const head_dim = config.headDim();
    const kv_dim = config.kvDim();
    const kv_mul = config.kvMul();
    const n_heads = config.n_heads;
    const n_layers = config.n_layers;
    const seq_len = config.seq_len;

    // Token embedding (always f32)
    const emb = weights.token_embedding[token * dim .. (token + 1) * dim];
    @memcpy(state.x, emb);

    for (0..n_layers) |l| {
        const ql = &qw.layers[l];

        // Attention RMSNorm (f32)
        rmsNorm(state.xb, state.x, weights.rms_att_weight[l * dim .. (l + 1) * dim]);

        // QKV projections — NATIVE QUANTIZED (the main speedup)
        matmulRaw(state.q, state.xb, ql.wq.raw, ql.wq.n_cols, ql.wq.tensor_type);
        matmulRaw(state.k, state.xb, ql.wk.raw, ql.wk.n_cols, ql.wk.tensor_type);
        matmulRaw(state.v, state.xb, ql.wv.raw, ql.wv.n_cols, ql.wv.tensor_type);

        // RoPE (same as f32 path)
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const q0 = q_head[i];
                const q1 = q_head[i + 1];
                q_head[i] = q0 * cos_a - q1 * sin_a;
                q_head[i + 1] = q0 * sin_a + q1 * cos_a;
            }
        }
        for (0..config.n_kv_heads) |h| {
            const k_head = state.k[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const k0 = k_head[i];
                const k1 = k_head[i + 1];
                k_head[i] = k0 * cos_a - k1 * sin_a;
                k_head[i + 1] = k0 * sin_a + k1 * cos_a;
            }
        }

        // KV cache (same)
        const cache_offset = l * seq_len * kv_dim + pos * kv_dim;
        @memcpy(state.key_cache[cache_offset .. cache_offset + kv_dim], state.k);
        @memcpy(state.value_cache[cache_offset .. cache_offset + kv_dim], state.v);

        // Attention (same — operates on f32 Q, cached K/V)
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            const att = state.att[h * seq_len .. h * seq_len + pos + 1];
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const k_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const k_vec = state.key_cache[k_offset .. k_offset + head_dim];
                var score: f32 = 0;
                for (0..head_dim) |d| score += q_head[d] * k_vec[d];
                att[t] = score / @sqrt(@as(f32, @floatFromInt(head_dim)));
            }
            softmax(att);
            const xb_head = state.xb[h * head_dim .. (h + 1) * head_dim];
            @memset(xb_head, 0);
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const v_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const v_vec = state.value_cache[v_offset .. v_offset + head_dim];
                for (0..head_dim) |d| xb_head[d] += att[t] * v_vec[d];
            }
        }

        // Output projection — NATIVE QUANTIZED
        matmulRaw(state.xb2, state.xb, ql.wo.raw, ql.wo.n_cols, ql.wo.tensor_type);
        for (0..dim) |i| state.x[i] += state.xb2[i];

        // FFN
        rmsNorm(state.xb, state.x, weights.rms_ffn_weight[l * dim .. (l + 1) * dim]);
        matmulRaw(state.hb, state.xb, ql.w1.raw, ql.w1.n_cols, ql.w1.tensor_type);
        matmulRaw(state.hb2, state.xb, ql.w3.raw, ql.w3.n_cols, ql.w3.tensor_type);

        // SwiGLU
        for (0..hidden_dim) |i| {
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + @exp(-state.hb[i]))) * state.hb2[i];
        }

        matmulRaw(state.xb, state.hb, ql.w2.raw, ql.w2.n_cols, ql.w2.tensor_type);
        for (0..dim) |i| state.x[i] += state.xb[i];
    }

    // Final RMSNorm (f32)
    rmsNorm(state.x, state.x, weights.rms_final_weight);

    // Classifier — quantized if available, else f32
    if (qw.raw_wcls) |raw| {
        matmulRaw(state.logits, state.x, raw, qw.wcls_cols, qw.wcls_type);
    } else {
        matmul(state.logits, state.x, weights.wcls, dim);
    }
}

// ── Model Loading ────────────────────────────────────────────────

pub const Model = struct {
    config: Config,
    weights: Weights,
    data: []align(4) const u8, // mmap'd or loaded file data

    pub fn load(path: []const u8, allocator: std.mem.Allocator) !Model {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const data = try allocator.alignedAlloc(u8, 4, file_size);
        const bytes_read = try file.readAll(data);
        if (bytes_read != file_size) return error.IncompleteRead;

        // Parse config (7 × i32 at start of file)
        const config_ints: *const [7]i32 = @ptrCast(@alignCast(data.ptr));
        const config = Config{
            .dim = @intCast(config_ints[0]),
            .hidden_dim = @intCast(config_ints[1]),
            .n_layers = @intCast(config_ints[2]),
            .n_heads = @intCast(config_ints[3]),
            .n_kv_heads = @intCast(config_ints[4]),
            .vocab_size = config_ints[5],
            .seq_len = @intCast(config_ints[6]),
        };

        const shared_weights = config.vocab_size > 0;
        const abs_vocab: u32 = @intCast(if (config.vocab_size < 0) -config.vocab_size else config.vocab_size);
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const n_layers = config.n_layers;
        const kv_dim = config.kvDim();

        // Parse weights — all contiguous f32 after the config
        const float_data: [*]const f32 = @ptrCast(@alignCast(data.ptr + 7 * 4));
        var offset: usize = 0;

        const token_embedding = float_data[offset .. offset + abs_vocab * dim];
        offset += abs_vocab * dim;

        const rms_att_weight = float_data[offset .. offset + n_layers * dim];
        offset += n_layers * dim;

        const wq = float_data[offset .. offset + n_layers * dim * dim];
        offset += n_layers * dim * dim;

        const wk = float_data[offset .. offset + n_layers * dim * kv_dim];
        offset += n_layers * dim * kv_dim;

        const wv = float_data[offset .. offset + n_layers * dim * kv_dim];
        offset += n_layers * dim * kv_dim;

        const wo = float_data[offset .. offset + n_layers * dim * dim];
        offset += n_layers * dim * dim;

        const rms_ffn_weight = float_data[offset .. offset + n_layers * dim];
        offset += n_layers * dim;

        const w1 = float_data[offset .. offset + n_layers * hidden_dim * dim];
        offset += n_layers * hidden_dim * dim;

        const w2 = float_data[offset .. offset + n_layers * dim * hidden_dim];
        offset += n_layers * dim * hidden_dim;

        const w3 = float_data[offset .. offset + n_layers * hidden_dim * dim];
        offset += n_layers * hidden_dim * dim;

        const rms_final_weight = float_data[offset .. offset + dim];
        offset += dim;

        // Skip freq_cis (not needed, we compute RoPE inline)
        offset += config.seq_len * config.headDim() / 2; // freq_cis_real
        offset += config.seq_len * config.headDim() / 2; // freq_cis_imag

        const wcls = if (shared_weights) token_embedding else float_data[offset .. offset + abs_vocab * dim];

        return .{
            .config = config,
            .weights = .{
                .token_embedding = token_embedding,
                .rms_att_weight = rms_att_weight,
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .wo = wo,
                .rms_ffn_weight = rms_ffn_weight,
                .w1 = w1,
                .w2 = w2,
                .w3 = w3,
                .rms_final_weight = rms_final_weight,
                .wcls = wcls,
            },
            .data = data,
        };
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

// ── Tokenizer ────────────────────────────────────────────────────

pub const Tokenizer = struct {
    vocab: [][]const u8,
    vocab_size: u32,
    data: []const u8,

    pub fn load(path: []const u8, vocab_size: u32, allocator: std.mem.Allocator) !Tokenizer {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const data = try allocator.alloc(u8, file_size);
        _ = try file.readAll(data);

        var vocab = try allocator.alloc([]const u8, vocab_size);

        // Skip max_token_length (first i32)
        var pos: usize = 4;
        for (0..vocab_size) |i| {
            if (pos + 4 > data.len) break;
            // score (f32) + len (i32) + string bytes
            pos += 4; // skip score
            const len: u32 = @as(*const u32, @ptrCast(@alignCast(data[pos..].ptr))).*;
            pos += 4;
            if (pos + len > data.len) break;
            vocab[i] = data[pos .. pos + len];
            pos += len;
        }

        return .{
            .vocab = vocab,
            .vocab_size = vocab_size,
            .data = data,
        };
    }

    pub fn decode(self: *const Tokenizer, token: u32) []const u8 {
        if (token >= self.vocab_size) return "<unk>";
        return self.vocab[token];
    }

    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.vocab);
        allocator.free(self.data);
    }
};

// ── W8A8 Weights and Forward ─────────────────────────────────────

pub const W8A8ModelWeights = struct {
    wq: w8a8.W8A8Weights,
    wk: w8a8.W8A8Weights,
    wv: w8a8.W8A8Weights,
    wo: w8a8.W8A8Weights,
    w1: w8a8.W8A8Weights,
    w2: w8a8.W8A8Weights,
    w3: w8a8.W8A8Weights,
    wcls: w8a8.W8A8Weights,

    pub fn fromWeights(wt: *const Weights, config: Config, allocator: std.mem.Allocator) !W8A8ModelWeights {
        const dim = config.dim;
        const hidden = config.hidden_dim;
        const kv = config.kvDim();
        const nl = config.n_layers;
        const vocab = @as(u32, @intCast(if (config.vocab_size < 0) -config.vocab_size else config.vocab_size));

        return .{
            .wq = try w8a8.W8A8Weights.fromF32(wt.wq, nl * dim, dim, allocator),
            .wk = try w8a8.W8A8Weights.fromF32(wt.wk, nl * kv, dim, allocator),
            .wv = try w8a8.W8A8Weights.fromF32(wt.wv, nl * kv, dim, allocator),
            .wo = try w8a8.W8A8Weights.fromF32(wt.wo, nl * dim, dim, allocator),
            .w1 = try w8a8.W8A8Weights.fromF32(wt.w1, nl * hidden, dim, allocator),
            .w2 = try w8a8.W8A8Weights.fromF32(wt.w2, nl * dim, hidden, allocator),
            .w3 = try w8a8.W8A8Weights.fromF32(wt.w3, nl * hidden, dim, allocator),
            .wcls = try w8a8.W8A8Weights.fromF32(wt.wcls, vocab, dim, allocator),
        };
    }

    pub fn deinit(self: *W8A8ModelWeights, allocator: std.mem.Allocator) void {
        self.wq.deinit(allocator);
        self.wk.deinit(allocator);
        self.wv.deinit(allocator);
        self.wo.deinit(allocator);
        self.w1.deinit(allocator);
        self.w2.deinit(allocator);
        self.w3.deinit(allocator);
        self.wcls.deinit(allocator);
    }
};

fn w8a8Matmul(out: []f32, x: []const f32, ww: *const w8a8.W8A8Weights, row_offset: usize, n_cols: usize) void {
    const n_rows = out.len;
    // Use NEON SDOT-accelerated matmul on ARM
    if (@import("builtin").cpu.arch == .aarch64) {
        neon.matmulNeonThreaded(
            ww.scales[row_offset .. row_offset + n_rows],
            ww.data[row_offset * n_cols ..],
            x,
            out,
            n_cols,
            n_rows,
        );
    } else {
        w8a8.matmulW8A8Threaded(
            ww.scales[row_offset .. row_offset + n_rows],
            ww.data[row_offset * n_cols ..],
            x,
            out,
            n_cols,
            n_rows,
        );
    }
}

pub fn forwardW8A8(config: Config, weights: *const Weights, ww: *const W8A8ModelWeights, state: *RunState, token: u32, pos: u32) void {
    const dim = config.dim;
    const hidden_dim = config.hidden_dim;
    const head_dim = config.headDim();
    const kv_dim = config.kvDim();
    const kv_mul = config.kvMul();
    const n_heads = config.n_heads;
    const n_layers = config.n_layers;
    const seq_len = config.seq_len;

    const emb = weights.token_embedding[token * dim .. (token + 1) * dim];
    @memcpy(state.x, emb);

    for (0..n_layers) |l| {
        rmsNorm(state.xb, state.x, weights.rms_att_weight[l * dim .. (l + 1) * dim]);

        // W8A8 matmuls — pure integer inner loop
        w8a8Matmul(state.q, state.xb, &ww.wq, l * dim, dim);
        w8a8Matmul(state.k, state.xb, &ww.wk, l * kv_dim, dim);
        w8a8Matmul(state.v, state.xb, &ww.wv, l * kv_dim, dim);

        // RoPE
        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const q0 = q_head[i];
                const q1 = q_head[i + 1];
                q_head[i] = q0 * cos_a - q1 * sin_a;
                q_head[i + 1] = q0 * sin_a + q1 * cos_a;
            }
        }
        for (0..config.n_kv_heads) |h| {
            const k_head = state.k[h * head_dim .. (h + 1) * head_dim];
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, config.rope_freq_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                const k0 = k_head[i];
                const k1 = k_head[i + 1];
                k_head[i] = k0 * cos_a - k1 * sin_a;
                k_head[i + 1] = k0 * sin_a + k1 * cos_a;
            }
        }

        const cache_offset = l * seq_len * kv_dim + pos * kv_dim;
        @memcpy(state.key_cache[cache_offset .. cache_offset + kv_dim], state.k);
        @memcpy(state.value_cache[cache_offset .. cache_offset + kv_dim], state.v);

        for (0..n_heads) |h| {
            const q_head = state.q[h * head_dim .. (h + 1) * head_dim];
            const att = state.att[h * seq_len .. h * seq_len + pos + 1];
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const k_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const k_vec = state.key_cache[k_offset .. k_offset + head_dim];
                var score: f32 = 0;
                for (0..head_dim) |d_i| score += q_head[d_i] * k_vec[d_i];
                att[t] = score / @sqrt(@as(f32, @floatFromInt(head_dim)));
            }
            softmax(att);
            const xb_head = state.xb[h * head_dim .. (h + 1) * head_dim];
            @memset(xb_head, 0);
            for (0..pos + 1) |t| {
                const kv_h = h / kv_mul;
                const v_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                const v_vec = state.value_cache[v_offset .. v_offset + head_dim];
                for (0..head_dim) |d_i| xb_head[d_i] += att[t] * v_vec[d_i];
            }
        }

        w8a8Matmul(state.xb2, state.xb, &ww.wo, l * dim, dim);
        for (0..dim) |i| state.x[i] += state.xb2[i];

        rmsNorm(state.xb, state.x, weights.rms_ffn_weight[l * dim .. (l + 1) * dim]);
        w8a8Matmul(state.hb, state.xb, &ww.w1, l * hidden_dim, dim);
        w8a8Matmul(state.hb2, state.xb, &ww.w3, l * hidden_dim, dim);

        for (0..hidden_dim) |i| {
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + @exp(-state.hb[i]))) * state.hb2[i];
        }

        w8a8Matmul(state.xb, state.hb, &ww.w2, l * dim, hidden_dim);
        for (0..dim) |i| state.x[i] += state.xb[i];
    }

    rmsNorm(state.x, state.x, weights.rms_final_weight);
    w8a8Matmul(state.logits, state.x, &ww.wcls, 0, dim);
}

// ── Sampling ─────────────────────────────────────────────────────

pub fn sample(logits: []f32, temperature: f32, rng: *std.Random) u32 {
    if (temperature <= 0.0) {
        // Greedy
        var best: usize = 0;
        for (1..logits.len) |i| {
            if (logits[i] > logits[best]) best = i;
        }
        return @intCast(best);
    }

    // Temperature scaling + softmax
    for (logits) |*v| v.* /= temperature;
    softmax(logits);

    // Sample from distribution
    var r = rng.float(f32);
    for (logits, 0..) |p, i| {
        r -= p;
        if (r <= 0) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

// ── Generate ─────────────────────────────────────────────────────

pub fn generate(
    model: *const Model,
    tokenizer_inst: *const Tokenizer,
    state: *RunState,
    prompt: []const u8,
    max_tokens: u32,
    temperature: f32,
    rng: *std.Random,
    writer: anytype,
) !void {
    _ = prompt; // TODO: encode prompt to tokens

    // Start with BOS token (1)
    var token: u32 = 1;
    var pos: u32 = 0;

    while (pos < max_tokens and pos < model.config.seq_len) {
        forward(model.config, &model.weights, state, token, pos);

        // Sample next token
        token = sample(state.logits, temperature, rng);

        // Decode and print
        const piece = tokenizer_inst.decode(token);
        try writer.writeAll(piece);

        pos += 1;
    }
}

// ── GGUF Tokenizer ──────────────────────────────────────────────────

pub const GGUFTokenizer = struct {
    vocab: []const []const u8,
    vocab_size: u32,

    pub fn decode(self: *const GGUFTokenizer, token: u32) []const u8 {
        if (token >= self.vocab_size) return "<unk>";
        return self.vocab[token];
    }
};

// ── GGUF Model ──────────────────────────────────────────────────────

/// A model loaded from GGUF format. Owns all dequantized weight data.
pub const GGUFModel = struct {
    config: Config,
    weights: Weights,

    // Owned allocations for dequantized weights
    token_embedding_data: []f32,
    rms_att_weight_data: []f32,
    wq_data: []f32,
    wk_data: []f32,
    wv_data: []f32,
    wo_data: []f32,
    rms_ffn_weight_data: []f32,
    w1_data: []f32, // ffn_gate
    w2_data: []f32, // ffn_down
    w3_data: []f32, // ffn_up
    rms_final_weight_data: []f32,
    wcls_data: ?[]f32, // null if tied with token_embedding

    // Raw quantized data pointers (for native quantized matmul)
    raw_wq: ?[]const u8 = null,
    raw_wk: ?[]const u8 = null,
    raw_wv: ?[]const u8 = null,
    raw_wo: ?[]const u8 = null,
    raw_w1: ?[]const u8 = null,
    raw_w2: ?[]const u8 = null,
    raw_w3: ?[]const u8 = null,
    raw_wcls: ?[]const u8 = null,
    quant_type: u32 = 0, // TensorType of the weight matrices

    // GGUF tokenizer
    vocab_strings: ?[][]const u8,
    gguf_file: ?gguf_mod.GGUFFile,

    allocator: std.mem.Allocator,

    pub fn load(path: []const u8, allocator: std.mem.Allocator) !GGUFModel {
        var gf = try gguf_mod.GGUFFile.open(path, allocator);
        errdefer gf.deinit();

        // Read config from metadata
        const dim = (gf.getMetadata("llama.embedding_length") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;
        const hidden_dim = (gf.getMetadata("llama.feed_forward_length") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;
        const n_layers = (gf.getMetadata("llama.block_count") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;
        const n_heads = (gf.getMetadata("llama.attention.head_count") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;
        const n_kv_heads = (gf.getMetadata("llama.attention.head_count_kv") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;
        const vocab_size = (gf.getMetadata("llama.vocab_size") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;
        const ctx_len = (gf.getMetadata("llama.context_length") orelse return error.MissingConfig).asU32() orelse return error.BadConfig;

        var rope_freq_base: f32 = 10000.0;
        if (gf.getMetadata("llama.rope.freq_base")) |rfb| {
            if (rfb.asF32()) |f| rope_freq_base = f;
        }

        const config = Config{
            .dim = dim,
            .hidden_dim = hidden_dim,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .vocab_size = @intCast(vocab_size),
            .seq_len = ctx_len,
            .rope_freq_base = rope_freq_base,
        };

        const kv_dim = config.kvDim();

        const stderr = std.io.getStdErr().writer();

        // Dequantize all weights from GGUF tensors
        // The GGUF tensor layout for llama is: blk.{layer}.{name}
        // Weight matrices in GGUF are [out_dim, in_dim] (row-major),
        // same as our matmul expects: w[i * n + j] for output i, input j.

        // Token embedding: [dim, vocab_size] -> we need [vocab_size * dim] flat, row per token
        try stderr.print("  Loading token_embd.weight...\n", .{});
        const token_embedding_data = try gf.dequantizeTensor(
            gf.getTensorInfo("token_embd.weight") orelse return error.MissingTensor,
            allocator,
        );
        errdefer allocator.free(token_embedding_data);

        // Per-layer weights: concatenate across layers
        try stderr.print("  Loading {d} layers...\n", .{n_layers});

        var rms_att = try allocator.alloc(f32, n_layers * dim);
        errdefer allocator.free(rms_att);
        var wq_all = try allocator.alloc(f32, n_layers * dim * dim);
        errdefer allocator.free(wq_all);
        var wk_all = try allocator.alloc(f32, n_layers * dim * kv_dim);
        errdefer allocator.free(wk_all);
        var wv_all = try allocator.alloc(f32, n_layers * dim * kv_dim);
        errdefer allocator.free(wv_all);
        var wo_all = try allocator.alloc(f32, n_layers * dim * dim);
        errdefer allocator.free(wo_all);
        var rms_ffn = try allocator.alloc(f32, n_layers * dim);
        errdefer allocator.free(rms_ffn);
        var w1_all = try allocator.alloc(f32, n_layers * hidden_dim * dim);
        errdefer allocator.free(w1_all);
        var w2_all = try allocator.alloc(f32, n_layers * dim * hidden_dim);
        errdefer allocator.free(w2_all);
        var w3_all = try allocator.alloc(f32, n_layers * hidden_dim * dim);
        errdefer allocator.free(w3_all);

        var name_buf: [64]u8 = undefined;

        for (0..n_layers) |l| {
            // attn_norm
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(rms_att[l * dim .. (l + 1) * dim], t);
            }
            // attn_q
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(wq_all[l * dim * dim .. (l + 1) * dim * dim], t);
            }
            // attn_k
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(wk_all[l * dim * kv_dim .. (l + 1) * dim * kv_dim], t);
            }
            // attn_v
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(wv_all[l * dim * kv_dim .. (l + 1) * dim * kv_dim], t);
            }
            // attn_output
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(wo_all[l * dim * dim .. (l + 1) * dim * dim], t);
            }
            // ffn_norm
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(rms_ffn[l * dim .. (l + 1) * dim], t);
            }
            // ffn_gate (w1)
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(w1_all[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim], t);
            }
            // ffn_down (w2)
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(w2_all[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], t);
            }
            // ffn_up (w3)
            {
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{l});
                const t = try gf.readTensorF32(name, allocator);
                defer allocator.free(t);
                @memcpy(w3_all[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim], t);
            }
        }

        // Final norm
        try stderr.print("  Loading output_norm.weight...\n", .{});
        const rms_final = try gf.readTensorF32("output_norm.weight", allocator);
        errdefer allocator.free(rms_final);

        // Output projection (wcls) — may be tied with token embedding
        var wcls_data: ?[]f32 = null;
        if (gf.getTensorInfo("output.weight")) |_| {
            try stderr.print("  Loading output.weight...\n", .{});
            wcls_data = try gf.readTensorF32("output.weight", allocator);
        }

        // Load tokenizer from GGUF metadata
        try stderr.print("  Loading tokenizer...\n", .{});
        var vocab_strings: ?[][]const u8 = null;
        vocab_strings = gf.readStringArray("tokenizer.ggml.tokens", allocator) catch null;

        const wcls_slice = if (wcls_data) |w| w else token_embedding_data;

        return .{
            .config = config,
            .weights = .{
                .token_embedding = token_embedding_data,
                .rms_att_weight = rms_att,
                .wq = wq_all,
                .wk = wk_all,
                .wv = wv_all,
                .wo = wo_all,
                .rms_ffn_weight = rms_ffn,
                .w1 = w1_all,
                .w2 = w2_all,
                .w3 = w3_all,
                .rms_final_weight = rms_final,
                .wcls = wcls_slice,
            },
            .token_embedding_data = token_embedding_data,
            .rms_att_weight_data = rms_att,
            .wq_data = wq_all,
            .wk_data = wk_all,
            .wv_data = wv_all,
            .wo_data = wo_all,
            .rms_ffn_weight_data = rms_ffn,
            .w1_data = w1_all,
            .w2_data = w2_all,
            .w3_data = w3_all,
            .rms_final_weight_data = rms_final,
            .wcls_data = wcls_data,
            .vocab_strings = vocab_strings,
            .gguf_file = gf,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GGUFModel) void {
        self.allocator.free(self.token_embedding_data);
        self.allocator.free(self.rms_att_weight_data);
        self.allocator.free(self.wq_data);
        self.allocator.free(self.wk_data);
        self.allocator.free(self.wv_data);
        self.allocator.free(self.wo_data);
        self.allocator.free(self.rms_ffn_weight_data);
        self.allocator.free(self.w1_data);
        self.allocator.free(self.w2_data);
        self.allocator.free(self.w3_data);
        self.allocator.free(self.rms_final_weight_data);
        if (self.wcls_data) |w| self.allocator.free(w);
        if (self.vocab_strings) |v| self.allocator.free(v);
        if (self.gguf_file) |*gf| {
            var gf_mut = gf.*;
            gf_mut.deinit();
        }
    }

    /// Get raw tensor data + metadata from the GGUF file by name.
    fn getQuantizedTensor(gf: *const gguf_mod.GGUFFile, name: []const u8) ?QuantizedTensor {
        const info = gf.getTensorInfo(name) orelse return null;
        const raw_bytes = (info.numElements() / info.tensor_type.blockSize()) * info.tensor_type.bytesPerBlock();
        const data_start = gf.data_offset + info.offset;
        if (data_start + raw_bytes > gf.file_data.len) return null;
        return .{
            .raw = gf.file_data[data_start .. data_start + raw_bytes],
            .tensor_type = @intFromEnum(info.tensor_type),
            .n_cols = @intCast(info.dims[0]), // GGUF dims[0] = columns (in_features)
        };
    }

    /// Build QuantizedWeights for native quantized forward pass.
    pub fn quantizedWeights(self: *const GGUFModel, allocator: std.mem.Allocator) !?QuantizedWeights {
        const gf = &(self.gguf_file orelse return null);

        const n_layers = self.config.n_layers;

        var layers = try allocator.alloc(QuantizedLayer, n_layers);

        for (0..n_layers) |l| {
            var buf_wq: [64]u8 = undefined;
            var buf_wk: [64]u8 = undefined;
            var buf_wv: [64]u8 = undefined;
            var buf_wo: [64]u8 = undefined;
            var buf_w1: [64]u8 = undefined;
            var buf_w2: [64]u8 = undefined;
            var buf_w3: [64]u8 = undefined;

            layers[l] = .{
                .wq = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_wq, "blk.{d}.attn_q.weight", .{l})) orelse return null,
                .wk = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_wk, "blk.{d}.attn_k.weight", .{l})) orelse return null,
                .wv = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_wv, "blk.{d}.attn_v.weight", .{l})) orelse return null,
                .wo = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_wo, "blk.{d}.attn_output.weight", .{l})) orelse return null,
                .w1 = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_w1, "blk.{d}.ffn_gate.weight", .{l})) orelse return null,
                .w2 = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_w2, "blk.{d}.ffn_down.weight", .{l})) orelse return null,
                .w3 = getQuantizedTensor(gf, try std.fmt.bufPrint(&buf_w3, "blk.{d}.ffn_up.weight", .{l})) orelse return null,
            };
        }

        const dim = self.config.dim;
        const wcls_qt = getQuantizedTensor(gf, "output.weight");

        return .{
            .layers = layers,
            .raw_wcls = if (wcls_qt) |qt| qt.raw else null,
            .wcls_type = if (wcls_qt) |qt| qt.tensor_type else 0,
            .wcls_cols = dim,
            .wcls_rows = @intCast(if (self.config.vocab_size < 0) -self.config.vocab_size else self.config.vocab_size),
        };
    }

    /// Create a Tokenizer from the GGUF-embedded vocabulary.
    pub fn tokenizer(self: *const GGUFModel) GGUFTokenizer {
        const vs = self.vocab_strings orelse &[_][]const u8{};
        return .{
            .vocab = vs,
            .vocab_size = @intCast(vs.len),
        };
    }
};
