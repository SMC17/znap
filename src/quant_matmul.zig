// Native quantized matrix-vector multiply kernels.
//
// These operate directly on quantized weight data without dequantizing to f32.
// This eliminates 4-8x memory bandwidth overhead.
//
// Supported formats (matching ggml-quants.c exactly):
//   Q4_0: 18 bytes/32 weights — f16 scale + 16 nibble bytes
//   Q5_0: 22 bytes/32 weights — f16 scale + 4 qh bytes + 16 nibble bytes
//   Q8_0: 34 bytes/32 weights — f16 scale + 32 int8 values
//   Q4_K: 144 bytes/256 weights — complex sub-block structure
//   Q6_K: 210 bytes/256 weights — 6-bit with sub-block scales

const std = @import("std");

fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

fn readU16LE(data: []const u8, offset: usize) u16 {
    return @as(u16, data[offset]) | (@as(u16, data[offset + 1]) << 8);
}

fn readU32LE(data: []const u8, offset: usize) u32 {
    return @as(u32, data[offset]) |
        (@as(u32, data[offset + 1]) << 8) |
        (@as(u32, data[offset + 2]) << 16) |
        (@as(u32, data[offset + 3]) << 24);
}

// ── Q4_0: 4-bit quantization ─────────────────────────────────────
// Block: [f16 scale (2)] [16 bytes nibbles (32 weights)] = 18 bytes

/// Q4_0 fused dot product — process 4 bytes at a time, pure @Vector.
pub fn dotQ4_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 32;
    const BB: usize = 18;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const d = f16ToF32(readU16LE(block, 0));
        const xb = x[block_idx * BS ..];

        var acc_lo: @Vector(4, f32) = @splat(0);
        var acc_hi: @Vector(4, f32) = @splat(0);

        comptime var j: usize = 0;
        inline while (j < 16) : (j += 4) {
            const lo = @Vector(4, f32){
                @floatFromInt(@as(i8, @intCast(block[2 + j] & 0xF)) - 8),
                @floatFromInt(@as(i8, @intCast(block[2 + j + 1] & 0xF)) - 8),
                @floatFromInt(@as(i8, @intCast(block[2 + j + 2] & 0xF)) - 8),
                @floatFromInt(@as(i8, @intCast(block[2 + j + 3] & 0xF)) - 8),
            };
            const hi = @Vector(4, f32){
                @floatFromInt(@as(i8, @intCast(block[2 + j] >> 4)) - 8),
                @floatFromInt(@as(i8, @intCast(block[2 + j + 1] >> 4)) - 8),
                @floatFromInt(@as(i8, @intCast(block[2 + j + 2] >> 4)) - 8),
                @floatFromInt(@as(i8, @intCast(block[2 + j + 3] >> 4)) - 8),
            };
            acc_lo += lo * @as(@Vector(4, f32), xb[j..][0..4].*);
            acc_hi += hi * @as(@Vector(4, f32), xb[j + 16 ..][0..4].*);
        }

        total += d * (@reduce(.Add, acc_lo) + @reduce(.Add, acc_hi));
    }
    return total;
}

// ── Q5_0: 5-bit quantization ─────────────────────────────────────
// Block: [f16 scale (2)] [u32 high bits (4)] [16 bytes nibbles] = 22 bytes

/// Q5_0 fused dequant+accumulate using vectorized byte extraction.
///
/// Key optimization: load 16 packed bytes as @Vector(16, u8), extract
/// all 16 low nibbles and 16 high nibbles simultaneously with vector
/// bitwise ops, convert to f32, multiply-accumulate.
pub fn dotQ5_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 32;
    const BB: usize = 22;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const d = f16ToF32(readU16LE(block, 0));
        const qh = readU32LE(block, 2);
        const xb = x[block_idx * BS ..];

        // Vector byte load: grab all 16 packed bytes at once
        const bytes: @Vector(16, u8) = block[6..22].*;
        const mask_lo: @Vector(16, u8) = @splat(0x0F);

        // Extract all 16 low nibbles simultaneously (vector AND)
        const lo_nibs = bytes & mask_lo;
        // Extract all 16 high nibbles simultaneously (vector shift)
        const hi_nibs = bytes >> @splat(4);

        // Now process in chunks of 4 for f32 conversion + accumulate
        var acc: @Vector(4, f32) = @splat(0);

        inline for (0..4) |chunk| {
            const off = chunk * 4;

            // Low nibble weights (indices off..off+3)
            const w_lo = @Vector(4, f32){
                @floatFromInt(@as(i16, lo_nibs[off] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off))) & 1)) << 4)) - 16),
                @floatFromInt(@as(i16, lo_nibs[off + 1] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 1))) & 1)) << 4)) - 16),
                @floatFromInt(@as(i16, lo_nibs[off + 2] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 2))) & 1)) << 4)) - 16),
                @floatFromInt(@as(i16, lo_nibs[off + 3] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 3))) & 1)) << 4)) - 16),
            };

            // High nibble weights (indices off+16..off+19)
            const w_hi = @Vector(4, f32){
                @floatFromInt(@as(i16, hi_nibs[off] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 16))) & 1)) << 4)) - 16),
                @floatFromInt(@as(i16, hi_nibs[off + 1] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 17))) & 1)) << 4)) - 16),
                @floatFromInt(@as(i16, hi_nibs[off + 2] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 18))) & 1)) << 4)) - 16),
                @floatFromInt(@as(i16, hi_nibs[off + 3] | (@as(u8, @intCast((qh >> @as(u5, @intCast(off + 19))) & 1)) << 4)) - 16),
            };

            acc += w_lo * @as(@Vector(4, f32), xb[off..][0..4].*);
            acc += w_hi * @as(@Vector(4, f32), xb[off + 16 ..][0..4].*);
        }

        total += d * @reduce(.Add, acc);
    }
    return total;
}

// ── Q8_0: 8-bit quantization ─────────────────────────────────────
// Block: [f16 scale (2)] [32 int8 values] = 34 bytes

/// Q8_0 fused dot product — process 4 int8 values at a time with @Vector.
pub fn dotQ8_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 32;
    const BB: usize = 34;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const d = f16ToF32(readU16LE(block, 0));
        const xb = x[block_idx * BS ..];

        // Process 4 int8 values at a time
        var acc0: @Vector(4, f32) = @splat(0);
        var acc1: @Vector(4, f32) = @splat(0);

        inline for (0..4) |chunk| {
            const off = chunk * 8;
            const q = @Vector(4, f32){
                @floatFromInt(@as(i8, @bitCast(block[2 + off]))),
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 1]))),
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 2]))),
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 3]))),
            };
            const q2 = @Vector(4, f32){
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 4]))),
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 5]))),
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 6]))),
                @floatFromInt(@as(i8, @bitCast(block[2 + off + 7]))),
            };
            acc0 += q * @as(@Vector(4, f32), xb[off..][0..4].*);
            acc1 += q2 * @as(@Vector(4, f32), xb[off + 4 ..][0..4].*);
        }

        total += d * (@reduce(.Add, acc0) + @reduce(.Add, acc1));
    }
    return total;
}

// ── Q6_K: 6-bit quantization ─────────────────────────────────────
// Block: 210 bytes per 256 weights
// Layout: [128 low nibbles] [64 high-2-bits] [16 scale int8s] [f16 d]

pub fn dotQ6_K(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 256;
    const BB: usize = 210;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const xb = x[block_idx * BS ..];

        const ql = block[0..128]; // Low 4 bits: 128 bytes
        const qh = block[128..192]; // High 2 bits: 64 bytes
        const sc = block[192..208]; // 16 int8 scales
        const d = f16ToF32(readU16LE(block, 208));

        // Matches dequantQ6_K in gguf.zig exactly.
        // Half 1: weights 0..127, using ql[0..63], qh[0..31]
        for (0..32) |l| {
            const is: usize = l / 16;
            const q1: i8 = @as(i8, @intCast((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
            const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
            const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
            const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;
            const s0: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 0])));
            const s2: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 2])));
            const s4: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 4])));
            const s6: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 6])));
            total += d * s0 * @as(f32, @floatFromInt(q1)) * xb[l + 0];
            total += d * s2 * @as(f32, @floatFromInt(q2)) * xb[l + 32];
            total += d * s4 * @as(f32, @floatFromInt(q3)) * xb[l + 64];
            total += d * s6 * @as(f32, @floatFromInt(q4)) * xb[l + 96];
        }

        // Half 2: weights 128..255, using ql[64..127], qh[32..63]
        for (0..32) |l| {
            const is: usize = l / 16;
            const q1: i8 = @as(i8, @intCast((ql[64 + l] & 0xF) | (((qh[32 + l] >> 0) & 3) << 4))) - 32;
            const q2: i8 = @as(i8, @intCast((ql[64 + l + 32] & 0xF) | (((qh[32 + l] >> 2) & 3) << 4))) - 32;
            const q3: i8 = @as(i8, @intCast((ql[64 + l] >> 4) | (((qh[32 + l] >> 4) & 3) << 4))) - 32;
            const q4: i8 = @as(i8, @intCast((ql[64 + l + 32] >> 4) | (((qh[32 + l] >> 6) & 3) << 4))) - 32;
            const s8: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 8])));
            const s10: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 10])));
            const s12: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 12])));
            const s14: f32 = @floatFromInt(@as(i8, @bitCast(sc[is + 14])));
            total += d * s8 * @as(f32, @floatFromInt(q1)) * xb[128 + l + 0];
            total += d * s10 * @as(f32, @floatFromInt(q2)) * xb[128 + l + 32];
            total += d * s12 * @as(f32, @floatFromInt(q3)) * xb[128 + l + 64];
            total += d * s14 * @as(f32, @floatFromInt(q4)) * xb[128 + l + 96];
        }
    }
    return total;
}

// ── Q4_K: 4-bit with sub-block scales ────────────────────────────
// Block: 144 bytes per 256 weights
// Layout: [f16 d (2)] [f16 dmin (2)] [12 scale bytes] [128 data bytes]

pub fn dotQ4_K(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 256;
    const BB: usize = 144;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const xb = x[block_idx * BS ..];

        const d = f16ToF32(readU16LE(block, 0));
        const dmin = f16ToF32(readU16LE(block, 2));
        const scales_raw = block[4..16]; // 12 bytes
        const qs = block[16..144]; // 128 bytes

        // Decode scales and mins using ggml's utmp bit manipulation.
        // This matches ggml_vec_dot_q4_K_q8_K_generic exactly:
        //   scales = (uint8_t*)&utmp[0] (8 bytes, utmp[0..1])
        //   mins   = (uint8_t*)&utmp[2] (8 bytes, utmp[2..3])
        const kmask1: u32 = 0x3f3f3f3f;
        const kmask2: u32 = 0x0f0f0f0f;
        const kmask3: u32 = 0x03030303;

        var utmp: [4]u32 = undefined;
        utmp[0] = readU32LE(scales_raw, 0);
        utmp[1] = readU32LE(scales_raw, 4);
        utmp[2] = readU32LE(scales_raw, 8);

        // Compute derived values before overwriting
        const new_utmp3 = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uaux = utmp[1] & kmask1;
        const new_utmp1 = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] = utmp[0] & kmask1;
        utmp[1] = new_utmp1;
        utmp[2] = uaux;
        utmp[3] = new_utmp3;

        // scales[0..7] = utmp[0..1] bytes, mins[0..7] = utmp[2..3] bytes
        const all_bytes: [16]u8 = @bitCast(utmp);
        const scales = all_bytes[0..8];
        const mins = all_bytes[8..16];

        // ggml layout: 4 groups of 64 weights.
        // Each group reads 32 bytes of qs. Within each group:
        //   - first 32 weights use low nibbles (qs[0..31] & 0xF), scale[is], min[is]
        //   - next 32 weights use high nibbles (qs[0..31] >> 4), scale[is+1], min[is+1]
        // q pointer advances by 32 each group.
        var q_off: usize = 0;
        var is: usize = 0;
        var w_off: usize = 0;
        while (w_off < 256) : (w_off += 64) {
            const d1 = d * @as(f32, @floatFromInt(scales[is]));
            const m1 = dmin * @as(f32, @floatFromInt(mins[is]));
            const d2 = d * @as(f32, @floatFromInt(scales[is + 1]));
            const m2 = dmin * @as(f32, @floatFromInt(mins[is + 1]));

            // 32 weights from low nibbles
            var sum1: f32 = 0;
            var sum_x1: f32 = 0;
            for (0..32) |l| {
                const nibble: f32 = @floatFromInt(qs[q_off + l] & 0xF);
                sum1 += nibble * xb[w_off + l];
                sum_x1 += xb[w_off + l];
            }
            total += d1 * sum1 - m1 * sum_x1;

            // 32 weights from high nibbles (same 32 bytes)
            var sum2: f32 = 0;
            var sum_x2: f32 = 0;
            for (0..32) |l| {
                const nibble: f32 = @floatFromInt(qs[q_off + l] >> 4);
                sum2 += nibble * xb[w_off + 32 + l];
                sum_x2 += xb[w_off + 32 + l];
            }
            total += d2 * sum2 - m2 * sum_x2;

            q_off += 32;
            is += 2;
        }
    }
    return total;
}

// ── Dispatch ─────────────────────────────────────────────────────

/// Compute a single row's dot product based on tensor type.
fn dotQuantizedRow(row_data: []const u8, x: []const f32, n_cols: usize, tensor_type: u32) f32 {
    return switch (tensor_type) {
        2 => dotQ4_0(row_data, x, n_cols),
        6 => dotQ5_0(row_data, x, n_cols),
        8 => dotQ8_0(row_data, x, n_cols),
        12 => dotQ4_K(row_data, x, n_cols),
        14 => dotQ6_K(row_data, x, n_cols),
        else => blk: {
            var sum: f32 = 0;
            const f32_row: [*]const f32 = @ptrCast(@alignCast(row_data.ptr));
            for (0..n_cols) |j| sum += f32_row[j] * x[j];
            break :blk sum;
        },
    };
}

const QuantWork = struct {
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    rb: usize,
    tensor_type: u32,
    start: usize,
    end: usize,
};

fn quantWorker(w: QuantWork) void {
    for (w.start..w.end) |i| {
        w.out[i] = dotQuantizedRow(w.raw_weights[i * w.rb ..][0..w.rb], w.x[0..w.n_cols], w.n_cols, w.tensor_type);
    }
}

/// Matrix-vector multiply dispatching to the correct quantized kernel.
/// Multithreaded for large output dimensions.
pub fn matmulQuantized(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
    tensor_type: u32,
) void {
    const rb = rowBytes(n_cols, tensor_type);

    // Multithread for large matmuls (FFN hidden dim = 1536)
    if (n_rows >= 192) {
        const n_threads = @min(std.Thread.getCpuCount() catch 1, 8);
        if (n_threads > 1) {
            const rows_per_thread = n_rows / n_threads;
            var threads: [8]std.Thread = undefined;
            var spawned: usize = 0;

            for (0..n_threads) |t| {
                const start = t * rows_per_thread;
                const end = if (t == n_threads - 1) n_rows else (t + 1) * rows_per_thread;
                threads[t] = std.Thread.spawn(.{}, quantWorker, .{QuantWork{
                    .raw_weights = raw_weights, .x = x, .out = out,
                    .n_cols = n_cols, .rb = rb, .tensor_type = tensor_type,
                    .start = start, .end = end,
                }}) catch {
                    quantWorker(.{ .raw_weights = raw_weights, .x = x, .out = out, .n_cols = n_cols, .rb = rb, .tensor_type = tensor_type, .start = start, .end = end });
                    continue;
                };
                spawned += 1;
            }
            for (0..spawned) |t| threads[t].join();
            return;
        }
    }

    for (0..n_rows) |i| {
        out[i] = dotQuantizedRow(raw_weights[i * rb ..][0..rb], x[0..n_cols], n_cols, tensor_type);
    }
}

pub fn rowBytes(n_cols: usize, tensor_type: u32) usize {
    return switch (tensor_type) {
        2 => (n_cols / 32) * 18, // Q4_0
        6 => (n_cols / 32) * 22, // Q5_0
        8 => (n_cols / 32) * 34, // Q8_0
        12 => (n_cols / 256) * 144, // Q4_K
        14 => (n_cols / 256) * 210, // Q6_K
        else => n_cols * 4, // F32
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "f16 conversion" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f16ToF32(0x3C00), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), f16ToF32(0x3800), 0.01);
}

test "Q4_0 dot product" {
    var block: [18]u8 = undefined;
    block[0] = 0x00;
    block[1] = 0x3C; // f16 1.0
    @memset(block[2..18], 0x99); // nibble 9 → value 9-8 = 1

    const x = [_]f32{1.0} ** 32;
    const result = dotQ4_0(&block, &x, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.5);
}

test "Q5_0 dot product" {
    // All zero high bits, nibbles = 16 (after subtracting 16 → 0)
    var block: [22]u8 = undefined;
    block[0] = 0x00;
    block[1] = 0x3C; // f16 1.0
    @memset(block[2..6], 0xFF); // all high bits set
    @memset(block[6..22], 0x00); // nibbles all 0

    // q5 = (0 | (1 << 4)) = 16, value = 16 - 16 = 0
    const x = [_]f32{1.0} ** 32;
    const result = dotQ5_0(&block, &x, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.5);
}

test "Q8_0 dot product" {
    var block: [34]u8 = undefined;
    block[0] = 0x00;
    block[1] = 0x38; // f16 0.5
    @memset(block[2..34], 2); // all int8 = 2

    const x = [_]f32{3.0} ** 32;
    const result = dotQ8_0(&block, &x, 32);
    // 0.5 * sum(2 * 3.0) = 0.5 * 192 = 96
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), result, 0.5);
}

test "matmulQuantized dispatches correctly" {
    var weights: [36]u8 = undefined;
    // 2 rows × 32 cols, Q4_0 (18 bytes per row)
    weights[0] = 0x00;
    weights[1] = 0x3C; // scale = 1.0
    @memset(weights[2..18], 0x99); // nibble 9 = value 1
    weights[18] = 0x00;
    weights[19] = 0x3C;
    @memset(weights[20..36], 0x77); // nibble 7 = value -1

    const x = [_]f32{1.0} ** 32;
    var out: [2]f32 = undefined;
    matmulQuantized(&weights, &x, &out, 32, 2, 2);

    try std.testing.expect(out[0] > 20);
    try std.testing.expect(out[1] < -20);
}

test "row bytes calculation" {
    try std.testing.expectEqual(@as(usize, 18), rowBytes(32, 2)); // Q4_0
    try std.testing.expectEqual(@as(usize, 22), rowBytes(32, 6)); // Q5_0
    try std.testing.expectEqual(@as(usize, 34), rowBytes(32, 8)); // Q8_0
    try std.testing.expectEqual(@as(usize, 144), rowBytes(256, 12)); // Q4_K
    try std.testing.expectEqual(@as(usize, 210), rowBytes(256, 14)); // Q6_K
}
