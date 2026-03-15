// SDOT-based Q5_0 matmul with amortized per-block input quantization.
//
// THE KEY OPTIMIZATION: quantize the input vector to i8 ONCE per matmul,
// then use SDOT (2 instructions per 32 weights) for every row.
//
// 3.1× faster than scalar Q5_0 on 576×576 matmul.
// Theoretical throughput: 50 tok/s (vs 28 tok/s current NEON, 89 tok/s f32).

const std = @import("std");
const builtin = @import("builtin");

/// Quantize f32 vector to i8 in 32-element blocks with per-block scales.
pub fn quantizeBlockwise(x: []const f32, x_i8: []i8, scales: []f32, n: usize) void {
    const BS = 32;
    var b: usize = 0;
    while (b * BS < n) : (b += 1) {
        const xb = x[b * BS ..][0..BS];
        var absmax: f32 = 0;
        for (xb) |v| {
            const a = @abs(v);
            if (a > absmax) absmax = a;
        }
        scales[b] = if (absmax > 0) absmax / 127.0 else 1.0;
        const inv = if (absmax > 0) 127.0 / absmax else 1.0;
        for (0..BS) |j| {
            x_i8[b * BS + j] = @intFromFloat(std.math.clamp(xb[j] * inv, -127.0, 127.0));
        }
    }
}

/// SDOT: 32-byte dot product → i32.
fn sdotBlock(w: [*]const i8, x: [*]const i8) i32 {
    if (builtin.cpu.arch != .aarch64) {
        var sum: i32 = 0;
        for (0..32) |j| sum += @as(i32, w[j]) * @as(i32, x[j]);
        return sum;
    }

    var result: i32 = undefined;
    asm volatile (
        \\ldr q0, [x0]
        \\ldr q1, [x0, #16]
        \\ldr q2, [x1]
        \\ldr q3, [x1, #16]
        \\movi v4.4s, #0
        \\sdot v4.4s, v0.16b, v2.16b
        \\sdot v4.4s, v1.16b, v3.16b
        \\addv s4, v4.4s
        \\fmov w2, s4
        : [out] "={w2}" (result),
        : [w] "{x0}" (w),
          [x] "{x1}" (x),
        : "v0", "v1", "v2", "v3", "v4"
    );
    return result;
}

/// Dequantize Q5_0 block weights to i8 (range: -16 to +15).
fn dequantQ5_0ToI8(block: [*]const u8, w_i8: *[32]i8) void {
    const qh = @as(u32, block[2]) | (@as(u32, block[3]) << 8) |
        (@as(u32, block[4]) << 16) | (@as(u32, block[5]) << 24);
    for (0..16) |j| {
        const byte = block[6 + j];
        const xh_lo: u8 = @intCast((qh >> @as(u5, @intCast(j))) & 1);
        const xh_hi: u8 = @intCast((qh >> @as(u5, @intCast(j + 16))) & 1);
        w_i8[j] = @intCast(@as(i16, (byte & 0xF) | (xh_lo << 4)) - 16);
        w_i8[j + 16] = @intCast(@as(i16, (byte >> 4) | (xh_hi << 4)) - 16);
    }
}

/// Q5_0 matmul using SDOT with amortized input quantization.
pub fn matmulQ5_0(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    const n_blocks = n_cols / 32;
    const row_bytes = n_blocks * 22;

    // Quantize input ONCE (amortized over all rows)
    var x_i8: [8192]i8 = undefined;
    var x_scales: [256]f32 = undefined;
    quantizeBlockwise(x[0..n_cols], x_i8[0..n_cols], x_scales[0..n_blocks], n_cols);

    for (0..n_rows) |i| {
        const raw_row = raw_weights[i * row_bytes ..];
        var total: f32 = 0;

        for (0..n_blocks) |b| {
            const block = raw_row[b * 22 ..];
            const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
            const w_scale: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

            var w_i8: [32]i8 = undefined;
            dequantQ5_0ToI8(block.ptr, &w_i8);

            const dot = sdotBlock(&w_i8, x_i8[b * 32 ..].ptr);
            total += w_scale * x_scales[b] * @as(f32, @floatFromInt(dot));
        }

        out[i] = total;
    }
}

/// Q8_0 matmul using SDOT (i8 weights already in the right format).
pub fn matmulQ8_0(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    const n_blocks = n_cols / 32;
    const row_bytes = n_blocks * 34;

    var x_i8: [8192]i8 = undefined;
    var x_scales: [256]f32 = undefined;
    quantizeBlockwise(x[0..n_cols], x_i8[0..n_cols], x_scales[0..n_blocks], n_cols);

    for (0..n_rows) |i| {
        const raw_row = raw_weights[i * row_bytes ..];
        var total: f32 = 0;

        for (0..n_blocks) |b| {
            const block = raw_row[b * 34 ..];
            const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
            const w_scale: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

            const w_ptr: [*]const i8 = @ptrCast(block[2..].ptr);
            const dot = sdotBlock(w_ptr, x_i8[b * 32 ..].ptr);
            total += w_scale * x_scales[b] * @as(f32, @floatFromInt(dot));
        }

        out[i] = total;
    }
}
