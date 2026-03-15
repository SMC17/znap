// Hand-rolled ARM NEON assembly kernels for quantized inference.
//
// These use SDOT (signed dot product) instructions that process
// 16 i8×i8 multiply-accumulates per cycle — 4× throughput vs f32.
//
// Only compiled on aarch64. Falls back to scalar on other archs.

const std = @import("std");
const builtin = @import("builtin");

/// SDOT-accelerated i8 dot product.
/// Computes dot(w[0..n], a[0..n]) using NEON SDOT instructions.
/// Processes 64 elements per loop iteration (4 × 16-byte SDOT).
pub fn dotI8Neon(w: [*]const i8, a: [*]const i8, n: usize) i32 {
    if (builtin.cpu.arch != .aarch64) {
        // Scalar fallback for non-ARM
        var sum: i32 = 0;
        for (0..n) |i| sum += @as(i32, w[i]) * @as(i32, a[i]);
        return sum;
    }

    var total: i32 = 0;
    var offset: usize = 0;

    // Main loop: 64 elements per iteration (4 × SDOT of 16 bytes)
    while (offset + 64 <= n) : (offset += 64) {
        var partial: i32 = undefined;
        const w_ptr = w + offset;
        const a_ptr = a + offset;

        asm volatile (
        // Load 64 bytes of weights into v0-v3
            \\ldp q0, q1, [x0]
            \\ldp q2, q3, [x0, #32]
            // Load 64 bytes of activations into v4-v7
            \\ldp q4, q5, [x1]
            \\ldp q6, q7, [x1, #32]
            // Zero accumulator
            \\movi v16.4s, #0
            // 4 × SDOT: 64 multiply-accumulates
            \\sdot v16.4s, v0.16b, v4.16b
            \\sdot v16.4s, v1.16b, v5.16b
            \\sdot v16.4s, v2.16b, v6.16b
            \\sdot v16.4s, v3.16b, v7.16b
            // Reduce 4 × i32 lanes to scalar
            \\addv s16, v16.4s
            \\fmov w0, s16
            : [out] "={w0}" (partial),
            : [wp] "{x0}" (w_ptr),
              [ap] "{x1}" (a_ptr),
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16"
        );

        total += partial;
    }

    // Process remaining 16 bytes at a time
    while (offset + 16 <= n) : (offset += 16) {
        var partial: i32 = undefined;
        const w_ptr = w + offset;
        const a_ptr = a + offset;

        asm volatile (
            \\ldr q0, [x0]
            \\ldr q1, [x1]
            \\movi v2.4s, #0
            \\sdot v2.4s, v0.16b, v1.16b
            \\addv s2, v2.4s
            \\fmov w0, s2
            : [out] "={w0}" (partial),
            : [wp] "{x0}" (w_ptr),
              [ap] "{x1}" (a_ptr),
            : "v0", "v1", "v2"
        );

        total += partial;
    }

    // Scalar remainder
    while (offset < n) : (offset += 1) {
        total += @as(i32, w[offset]) * @as(i32, a[offset]);
    }

    return total;
}

/// NEON-accelerated W8A8 matrix-vector multiply.
/// weight_scales: [n_rows]f32 per-row scales
/// weight_i8: [n_rows * n_cols]i8 packed weights
/// x: [n_cols]f32 input (quantized to i8 internally)
/// out: [n_rows]f32 output
pub fn matmulNeon(
    weight_scales: []const f32,
    weight_i8: []const i8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    // Quantize input to i8 once
    var x_i8: [8192]i8 = undefined;
    var absmax: f32 = 0;
    for (x[0..n_cols]) |v| {
        const a = @abs(v);
        if (a > absmax) absmax = a;
    }
    const x_scale = if (absmax > 0) absmax / 127.0 else 1.0;
    const inv_scale = if (absmax > 0) 127.0 / absmax else 1.0;
    for (0..n_cols) |i| {
        x_i8[i] = @intFromFloat(std.math.clamp(x[i] * inv_scale, -127.0, 127.0));
    }

    // Compute each output row using NEON SDOT
    for (0..n_rows) |i| {
        const w_row: [*]const i8 = @ptrCast(weight_i8[i * n_cols ..].ptr);
        const a_row: [*]const i8 = @ptrCast(&x_i8);
        const i32_dot = dotI8Neon(w_row, a_row, n_cols);
        out[i] = @as(f32, @floatFromInt(i32_dot)) * weight_scales[i] * x_scale;
    }
}

/// Multithreaded NEON matmul.
pub fn matmulNeonThreaded(
    weight_scales: []const f32,
    weight_i8: []const i8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    // Quantize input once (shared across all threads)
    var x_i8: [8192]i8 = undefined;
    var absmax: f32 = 0;
    for (x[0..n_cols]) |v| {
        const a = @abs(v);
        if (a > absmax) absmax = a;
    }
    const x_scale = if (absmax > 0) absmax / 127.0 else 1.0;
    const inv = if (absmax > 0) 127.0 / absmax else 1.0;
    for (0..n_cols) |i| {
        x_i8[i] = @intFromFloat(std.math.clamp(x[i] * inv, -127.0, 127.0));
    }

    if (n_rows < 256) {
        for (0..n_rows) |i| {
            const w: [*]const i8 = @ptrCast(weight_i8[i * n_cols ..].ptr);
            const a: [*]const i8 = @ptrCast(&x_i8);
            out[i] = @as(f32, @floatFromInt(dotI8Neon(w, a, n_cols))) * weight_scales[i] * x_scale;
        }
        return;
    }

    const n_threads = @min(std.Thread.getCpuCount() catch 1, 8);
    const Work = struct { ws: []const f32, wi8: []const i8, xi8: *const [8192]i8, xs: f32, o: []f32, nc: usize, start: usize, end: usize };

    const worker = struct {
        fn f(w: Work) void {
            for (w.start..w.end) |i| {
                const wp: [*]const i8 = @ptrCast(w.wi8[i * w.nc ..].ptr);
                const ap: [*]const i8 = @ptrCast(w.xi8);
                w.o[i] = @as(f32, @floatFromInt(dotI8Neon(wp, ap, w.nc))) * w.ws[i] * w.xs;
            }
        }
    }.f;

    const rpt = n_rows / n_threads;
    var threads: [8]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..n_threads) |t| {
        const s = t * rpt;
        const e = if (t == n_threads - 1) n_rows else (t + 1) * rpt;
        threads[t] = std.Thread.spawn(.{}, worker, .{Work{
            .ws = weight_scales, .wi8 = weight_i8, .xi8 = &x_i8,
            .xs = x_scale, .o = out, .nc = n_cols, .start = s, .end = e,
        }}) catch {
            worker(.{ .ws = weight_scales, .wi8 = weight_i8, .xi8 = &x_i8, .xs = x_scale, .o = out, .nc = n_cols, .start = s, .end = e });
            continue;
        };
        spawned += 1;
    }
    for (0..spawned) |t| threads[t].join();
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "SDOT dot product basic" {
    const w = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, -1, -1, -1, -1 };
    const a = [_]i8{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const result = dotI8Neon(&w, &a, 16);
    // (1+2+3+4) + (5+6+7+8) + (1+1+1+1) + (-1-1-1-1) = 36
    try std.testing.expectEqual(@as(i32, 36), result);
}

test "SDOT dot product 64 elements" {
    var w: [64]i8 = undefined;
    var a: [64]i8 = undefined;
    for (0..64) |i| {
        w[i] = 1;
        a[i] = 2;
    }
    const result = dotI8Neon(&w, &a, 64);
    try std.testing.expectEqual(@as(i32, 128), result);
}

test "SDOT dot product 576 elements" {
    var w: [576]i8 = undefined;
    var a: [576]i8 = undefined;
    for (0..576) |i| {
        w[i] = 1;
        a[i] = 1;
    }
    const result = dotI8Neon(&w, &a, 576);
    try std.testing.expectEqual(@as(i32, 576), result);
}

test "NEON matmul basic" {
    var w_scales = [_]f32{ 1.0 / 127.0, 1.0 / 127.0 };
    var w_i8: [2 * 16]i8 = undefined;
    @memset(&w_i8, 0);
    w_i8[0] = 127; // row 0: [127, 0, 0, ...]
    w_i8[16 + 1] = 127; // row 1: [0, 127, 0, ...]

    var x: [16]f32 = undefined;
    @memset(&x, 0);
    x[0] = 1.0;
    x[1] = 0.5;

    var out: [2]f32 = undefined;
    matmulNeon(&w_scales, &w_i8, &x, &out, 16, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1], 0.05);
}
