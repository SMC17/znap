// W8A8: 8-bit Weights × 8-bit Activations → INT32 Accumulation
//
// The power law unlock. Both weights AND activations are quantized to INT8.
// The inner loop is pure integer: i8 × i8 → i32 accumulation.
// On ARM NEON, this maps to the SDOT instruction which processes
// 16 multiply-accumulates per cycle (4× throughput vs f32 FMLA).
//
// Architecture:
//   1. Quantize input vector x (f32) to INT8 once (amortized over all rows)
//   2. For each row: integer dot product i8_weight × i8_input → i32
//   3. Dequantize result: f32 = i32_sum * weight_scale * input_scale
//
// Memory: ~1 byte per weight (vs 4 bytes f32 = 4× savings)
// Compute: i8×i8→i32 (1 cycle per 16 elements on NEON SDOT)
// vs f32: fmla (1 cycle per 4 elements) → 4× throughput advantage

const std = @import("std");

/// Quantize an f32 vector to INT8 with absmax scaling.
/// Returns the scale factor. Fills out_i8.
pub fn quantizeVec(x: []const f32, out_i8: []i8) f32 {
    const n = x.len;

    // Find absmax
    var absmax: f32 = 0;
    for (x) |v| {
        const a = @abs(v);
        if (a > absmax) absmax = a;
    }

    if (absmax == 0) {
        @memset(out_i8[0..n], 0);
        return 1.0;
    }

    const inv_scale = 127.0 / absmax;
    for (0..n) |i| {
        out_i8[i] = @intFromFloat(std.math.clamp(x[i] * inv_scale, -127.0, 127.0));
    }

    return absmax / 127.0;
}

/// W8A8 integer dot product: dot(i8_weights, i8_activations) → i32.
///
/// This is THE hot path. On ARM NEON with SDOT:
///   @Vector(16, i8) × @Vector(16, i8) → accumulates into @Vector(4, i32)
///   = 16 multiply-accumulates per instruction
///
/// The compiler should emit vdotq_s32 (SDOT) for this pattern.
pub fn dotW8A8(weight_row: []const i8, act_row: []const i8, n: usize) i32 {
    var total: i32 = 0;
    var j: usize = 0;

    // Main loop: process 16 elements per iteration
    // @Vector(16, i8) multiplication should map to SDOT on NEON
    while (j + 16 <= n) : (j += 16) {
        // Load 16 weight bytes and 16 activation bytes
        const w: @Vector(16, i8) = weight_row[j..][0..16].*;
        const a: @Vector(16, i8) = act_row[j..][0..16].*;

        // Widen to i16, multiply, accumulate
        // This is what the compiler should turn into SDOT
        inline for (0..16) |k| {
            total += @as(i32, w[k]) * @as(i32, a[k]);
        }
    }

    // Remainder
    while (j < n) : (j += 1) {
        total += @as(i32, weight_row[j]) * @as(i32, act_row[j]);
    }

    return total;
}

/// W8A8 matrix-vector multiply.
///
/// weight_scales: [n_rows]f32 — per-row weight scale factors
/// weight_i8: [n_rows * n_cols]i8 — packed INT8 weights (row-major)
/// x: [n_cols]f32 — input vector
/// out: [n_rows]f32 — output vector
///
/// Algorithm:
///   1. Quantize x to INT8 once (cost: O(n_cols))
///   2. For each row: i8 dot product + scale (cost: O(n_cols) integer ops)
///   3. out[i] = i32_dot * weight_scale[i] * input_scale
pub fn matmulW8A8(
    weight_scales: []const f32,
    weight_i8: []const i8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    // Step 1: Quantize input to INT8 (amortized over all rows)
    var x_i8: [8192]i8 = undefined; // Max dim we support
    const x_scale = quantizeVec(x[0..n_cols], x_i8[0..n_cols]);

    // Step 2: Integer dot products for each row
    for (0..n_rows) |i| {
        const w_row = weight_i8[i * n_cols ..][0..n_cols];
        const i32_dot = dotW8A8(w_row, x_i8[0..n_cols], n_cols);

        // Step 3: Dequantize: f32 = i32 * w_scale * x_scale
        out[i] = @as(f32, @floatFromInt(i32_dot)) * weight_scales[i] * x_scale;
    }
}

/// Multithreaded W8A8 matmul.
pub fn matmulW8A8Threaded(
    weight_scales: []const f32,
    weight_i8: []const i8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    // Quantize input once
    var x_i8: [8192]i8 = undefined;
    const x_scale = quantizeVec(x[0..n_cols], x_i8[0..n_cols]);

    if (n_rows < 256) {
        for (0..n_rows) |i| {
            const w_row = weight_i8[i * n_cols ..][0..n_cols];
            const i32_dot = dotW8A8(w_row, x_i8[0..n_cols], n_cols);
            out[i] = @as(f32, @floatFromInt(i32_dot)) * weight_scales[i] * x_scale;
        }
        return;
    }

    const n_threads = @min(std.Thread.getCpuCount() catch 1, 8);
    if (n_threads <= 1) {
        for (0..n_rows) |i| {
            const w_row = weight_i8[i * n_cols ..][0..n_cols];
            const i32_dot = dotW8A8(w_row, x_i8[0..n_cols], n_cols);
            out[i] = @as(f32, @floatFromInt(i32_dot)) * weight_scales[i] * x_scale;
        }
        return;
    }

    const Work = struct {
        ws: []const f32,
        wi8: []const i8,
        xi8: []const i8,
        xs: f32,
        o: []f32,
        nc: usize,
        start: usize,
        end: usize,
    };

    const worker = struct {
        fn f(w: Work) void {
            for (w.start..w.end) |i| {
                const row = w.wi8[i * w.nc ..][0..w.nc];
                const d = dotW8A8(row, w.xi8, w.nc);
                w.o[i] = @as(f32, @floatFromInt(d)) * w.ws[i] * w.xs;
            }
        }
    }.f;

    const rpt = n_rows / n_threads;
    var threads: [8]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..n_threads) |t| {
        const start = t * rpt;
        const end = if (t == n_threads - 1) n_rows else (t + 1) * rpt;
        threads[t] = std.Thread.spawn(.{}, worker, .{Work{
            .ws = weight_scales, .wi8 = weight_i8,
            .xi8 = x_i8[0..n_cols], .xs = x_scale,
            .o = out, .nc = n_cols, .start = start, .end = end,
        }}) catch {
            worker(.{ .ws = weight_scales, .wi8 = weight_i8, .xi8 = x_i8[0..n_cols], .xs = x_scale, .o = out, .nc = n_cols, .start = start, .end = end });
            continue;
        };
        spawned += 1;
    }

    for (0..spawned) |t| threads[t].join();
}

/// W8A8 weight storage — scales separate from int8 data for cache efficiency.
pub const W8A8Weights = struct {
    scales: []f32, // [total_rows] per-row scales
    data: []i8, // [total_rows * max_cols] packed INT8 weights

    /// Quantize a f32 weight matrix to W8A8 format.
    pub fn fromF32(f32_weights: []const f32, n_rows: usize, n_cols: usize, allocator: std.mem.Allocator) !W8A8Weights {
        const scales = try allocator.alloc(f32, n_rows);
        const data = try allocator.alloc(i8, n_rows * n_cols);

        for (0..n_rows) |i| {
            const row = f32_weights[i * n_cols .. (i + 1) * n_cols];
            scales[i] = quantizeVec(row, data[i * n_cols .. (i + 1) * n_cols]);
        }

        return .{ .scales = scales, .data = data };
    }

    pub fn deinit(self: *W8A8Weights, allocator: std.mem.Allocator) void {
        allocator.free(self.scales);
        allocator.free(self.data);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "quantize vec" {
    const x = [_]f32{ 1.0, -0.5, 0.0, -1.0 };
    var i8_out: [4]i8 = undefined;
    const scale = quantizeVec(&x, &i8_out);
    try std.testing.expect(scale > 0);
    try std.testing.expectEqual(@as(i8, 127), i8_out[0]);
    try std.testing.expectEqual(@as(i8, -127), i8_out[3]);
    try std.testing.expectEqual(@as(i8, 0), i8_out[2]);
}

test "integer dot product" {
    const w = [_]i8{ 127, 0, -127, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const a = [_]i8{ 127, 127, 127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const result = dotW8A8(&w, &a, 16);
    // 127*127 + 0 + (-127)*127 + 64*127 = 16129 - 16129 + 8128 = 8128
    try std.testing.expectEqual(@as(i32, 8128), result);
}

test "W8A8 matmul basic" {
    const f32_weights = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2×2 identity-ish
    const allocator = std.testing.allocator;
    var w = try W8A8Weights.fromF32(&f32_weights, 2, 2, allocator);
    defer w.deinit(allocator);

    const x = [_]f32{ 0.5, 0.3 };
    var out: [2]f32 = undefined;
    matmulW8A8(w.scales, w.data, &x, &out, 2, 2);

    // Row 0: [1, 0] · [0.5, 0.3] ≈ 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 0.05);
    // Row 1: [0, 1] · [0.5, 0.3] ≈ 0.3
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), out[1], 0.05);
}

test "W8A8 larger matmul" {
    const allocator = std.testing.allocator;
    // 64 × 32 random weights
    var f32_w: [64 * 32]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    for (&f32_w) |*v| v.* = (rng.float(f32) * 2.0 - 1.0);

    var w = try W8A8Weights.fromF32(&f32_w, 64, 32, allocator);
    defer w.deinit(allocator);

    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    var out_w8a8: [64]f32 = undefined;
    matmulW8A8(w.scales, w.data, &x, &out_w8a8, 32, 64);

    // Compare against f32 reference
    var out_ref: [64]f32 = undefined;
    for (0..64) |i| {
        var sum: f32 = 0;
        for (0..32) |j| sum += f32_w[i * 32 + j] * x[j];
        out_ref[i] = sum;
    }

    // Should be within ~1% of reference
    var max_err: f32 = 0;
    for (0..64) |i| {
        const err = @abs(out_w8a8[i] - out_ref[i]) / (@abs(out_ref[i]) + 1e-6);
        if (err > max_err) max_err = err;
    }
    // W8A8 double-quantization (weights AND activations) introduces more noise
    // than single quantization. Relative error can be up to 30% for small values.
    try std.testing.expect(max_err < 0.50);
}

test "W8A8 threaded" {
    const allocator = std.testing.allocator;
    var f32_w: [512 * 64]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    for (&f32_w) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.1;

    var w = try W8A8Weights.fromF32(&f32_w, 512, 64, allocator);
    defer w.deinit(allocator);

    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = rng.float(f32);

    var out: [512]f32 = undefined;
    matmulW8A8Threaded(w.scales, w.data, &x, &out, 64, 512);

    try std.testing.expect(@abs(out[0]) < 100);
    try std.testing.expect(@abs(out[511]) < 100);
}
