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

pub fn dotQ4_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 32;
    const BB: usize = 18;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const d = f16ToF32(readU16LE(block, 0));
        const xb = x[block_idx * BS ..];

        var sum: f32 = 0;
        for (0..16) |j| {
            const byte = block[2 + j];
            const lo = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0xF)) - 8));
            const hi = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8));
            sum += lo * xb[j * 2] + hi * xb[j * 2 + 1];
        }
        total += d * sum;
    }
    return total;
}

// ── Q5_0: 5-bit quantization ─────────────────────────────────────
// Block: [f16 scale (2)] [u32 high bits (4)] [16 bytes nibbles] = 22 bytes

pub fn dotQ5_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 32;
    const BB: usize = 22;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const d = f16ToF32(readU16LE(block, 0));
        const qh = readU32LE(block, 2); // 32 high bits
        const xb = x[block_idx * BS ..];

        var sum_qx: f32 = 0;
        var sum_x: f32 = 0;

        for (0..16) |j| {
            const byte = block[6 + j];
            const lo_nib: u8 = byte & 0xF;
            const hi_nib: u8 = byte >> 4;

            // Extract high bits
            const qh_lo: u8 = @intCast((qh >> @intCast(j * 2)) & 1);
            const qh_hi: u8 = @intCast((qh >> @intCast(j * 2 + 1)) & 1);

            // 5-bit values
            const q5_lo: f32 = @floatFromInt(@as(u8, lo_nib | (qh_lo << 4)));
            const q5_hi: f32 = @floatFromInt(@as(u8, hi_nib | (qh_hi << 4)));

            sum_qx += q5_lo * xb[j * 2] + q5_hi * xb[j * 2 + 1];
            sum_x += xb[j * 2] + xb[j * 2 + 1];
        }
        total += d * (sum_qx - 16.0 * sum_x);
    }
    return total;
}

// ── Q8_0: 8-bit quantization ─────────────────────────────────────
// Block: [f16 scale (2)] [32 int8 values] = 34 bytes

pub fn dotQ8_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const BS: usize = 32;
    const BB: usize = 34;
    var total: f32 = 0;

    var block_idx: usize = 0;
    while (block_idx * BS < n) : (block_idx += 1) {
        const block = raw_row[block_idx * BB ..][0..BB];
        const d = f16ToF32(readU16LE(block, 0));
        const xb = x[block_idx * BS ..];

        var sum: f32 = 0;
        for (0..32) |j| {
            const q: i8 = @bitCast(block[2 + j]);
            sum += @as(f32, @floatFromInt(q)) * xb[j];
        }
        total += d * sum;
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

        const ql = block[0..128]; // Low 4 bits
        const qh = block[128..192]; // High 2 bits
        const sc = block[192..208]; // 16 int8 scales
        const d = f16ToF32(readU16LE(block, 208));

        // 16 sub-blocks of 16 weights each
        for (0..16) |sb| {
            const scale: f32 = @floatFromInt(@as(i8, @bitCast(sc[sb])));
            var sum: f32 = 0;

            for (0..16) |j| {
                const idx = sb * 16 + j;
                const ql_byte_idx = idx / 2;
                const ql_nibble: u8 = if (idx % 2 == 0) ql[ql_byte_idx] & 0xF else ql[ql_byte_idx] >> 4;
                const qh_byte_idx = idx / 4;
                const qh_shift: u3 = @intCast((idx % 4) * 2);
                const qh_bits: u8 = @intCast((qh[qh_byte_idx] >> qh_shift) & 3);

                const q6: i8 = @intCast(@as(i16, ql_nibble | (qh_bits << 4)) - 32);
                sum += @as(f32, @floatFromInt(q6)) * xb[idx];
            }
            total += d * scale * sum;
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
        const scales_raw = block[4..16];
        const qs = block[16..144];

        // Extract 6-bit scales and mins from 12 bytes
        // ggml packs them as: bytes 0-3 have scales[0:4] & 63 and mins[0:4] & 63
        // bytes 4-7 have scales[4:8] & 63, bytes 8-11 have high 2 bits
        var sc: [8]u8 = undefined;
        var mn: [8]u8 = undefined;

        for (0..4) |i| {
            sc[i] = scales_raw[i] & 63;
            mn[i] = scales_raw[i + 4] & 63;
        }
        for (0..4) |i| {
            const high_sc = (scales_raw[8 + i] >> 0) & 0x03;
            const high_mn = (scales_raw[8 + i] >> 2) & 0x03;
            sc[i] |= high_sc << 6;
            mn[i] |= high_mn << 6;

            const high_sc2 = (scales_raw[8 + i] >> 4) & 0x03;
            const high_mn2 = (scales_raw[8 + i] >> 6) & 0x03;
            sc[i + 4] = (scales_raw[i + 4] >> 6) | (high_sc2 << 2);
            mn[i + 4] = (scales_raw[i] >> 6) | (high_mn2 << 2);
        }

        // 8 sub-blocks of 32 weights
        for (0..8) |sb| {
            const d_sc = d * @as(f32, @floatFromInt(sc[sb]));
            const d_mn = dmin * @as(f32, @floatFromInt(mn[sb]));
            const data_off = sb * 16;

            var sum_qx: f32 = 0;
            var sum_x: f32 = 0;
            for (0..16) |j| {
                const byte = qs[data_off + j];
                const lo: f32 = @floatFromInt(byte & 0xF);
                const hi: f32 = @floatFromInt(byte >> 4);
                sum_qx += lo * xb[sb * 32 + j * 2] + hi * xb[sb * 32 + j * 2 + 1];
                sum_x += xb[sb * 32 + j * 2] + xb[sb * 32 + j * 2 + 1];
            }
            total += d_sc * sum_qx - d_mn * sum_x;
        }
    }
    return total;
}

// ── Dispatch ─────────────────────────────────────────────────────

/// Matrix-vector multiply dispatching to the correct quantized kernel.
/// Tensor types: 0=F32, 1=F16, 2=Q4_0, 6=Q5_0, 7=Q5_1, 8=Q8_0, 12=Q4_K, 14=Q6_K
pub fn matmulQuantized(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
    tensor_type: u32,
) void {
    const row_bytes = rowBytes(n_cols, tensor_type);

    for (0..n_rows) |i| {
        const row_data = raw_weights[i * row_bytes ..][0..row_bytes];
        out[i] = switch (tensor_type) {
            2 => dotQ4_0(row_data, x[0..n_cols], n_cols),
            6 => dotQ5_0(row_data, x[0..n_cols], n_cols),
            8 => dotQ8_0(row_data, x[0..n_cols], n_cols),
            12 => dotQ4_K(row_data, x[0..n_cols], n_cols),
            14 => dotQ6_K(row_data, x[0..n_cols], n_cols),
            else => blk: {
                // F32 fallback
                var sum: f32 = 0;
                const f32_row: [*]const f32 = @ptrCast(@alignCast(row_data.ptr));
                for (0..n_cols) |j| sum += f32_row[j] * x[j];
                break :blk sum;
            },
        };
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
