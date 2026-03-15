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

pub const Config = struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: i32, // negative = shared weights with token embedding
    seq_len: u32,

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

fn matmul(out: []f32, x: []const f32, w: []const f32, n: usize) void {
    const d = out.len;
    const VEC_SIZE = 4; // NEON: 4 × f32

    for (0..d) |i| {
        const row = w[i * n ..][0..n];

        // 4-way unrolled SIMD accumulation
        var acc0: @Vector(VEC_SIZE, f32) = @splat(0);
        var acc1: @Vector(VEC_SIZE, f32) = @splat(0);
        var acc2: @Vector(VEC_SIZE, f32) = @splat(0);
        var acc3: @Vector(VEC_SIZE, f32) = @splat(0);
        var j: usize = 0;

        // Main loop: 16 elements per iteration (4 vectors × 4 floats)
        while (j + 16 <= n) : (j += 16) {
            acc0 += @as(@Vector(VEC_SIZE, f32), row[j..][0..4].*) * @as(@Vector(VEC_SIZE, f32), x[j..][0..4].*);
            acc1 += @as(@Vector(VEC_SIZE, f32), row[j + 4 ..][0..4].*) * @as(@Vector(VEC_SIZE, f32), x[j + 4 ..][0..4].*);
            acc2 += @as(@Vector(VEC_SIZE, f32), row[j + 8 ..][0..4].*) * @as(@Vector(VEC_SIZE, f32), x[j + 8 ..][0..4].*);
            acc3 += @as(@Vector(VEC_SIZE, f32), row[j + 12 ..][0..4].*) * @as(@Vector(VEC_SIZE, f32), x[j + 12 ..][0..4].*);
        }

        // Reduce 4 accumulators
        var sum = @reduce(.Add, acc0 + acc1 + acc2 + acc3);

        // Handle remaining elements with single vector
        while (j + VEC_SIZE <= n) : (j += VEC_SIZE) {
            const rv: @Vector(VEC_SIZE, f32) = row[j..][0..VEC_SIZE].*;
            const xv: @Vector(VEC_SIZE, f32) = x[j..][0..VEC_SIZE].*;
            sum += @reduce(.Add, rv * xv);
        }

        // Scalar remainder
        while (j < n) : (j += 1) {
            sum += row[j] * x[j];
        }

        out[i] = sum;
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
                const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
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
                const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
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
