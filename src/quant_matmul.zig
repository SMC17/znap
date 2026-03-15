// Native quantized matrix-vector multiply kernels.
//
// These operate directly on quantized weight data without dequantizing to f32.
// This eliminates the memory bandwidth overhead of expanding 4-bit weights
// to 32-bit floats, and removes the dequantization compute from the hot path.
//
// Q4_K format: 144 bytes per block of 256 weights
//   - d (f16): master scale
//   - dmin (f16): minimum offset
//   - scales (12 bytes): 4-bit sub-block scales + mins for 8 sub-blocks of 32
//   - data (128 bytes): 4-bit weight nibbles (256 weights, 2 per byte)
//
// The key insight: we never expand to f32. Instead:
//   dot(Q4_K_row, f32_vec) = sum_over_blocks(scale * sum(nibble * x) - min * sum(x))
// Each block's contribution is computed with integer arithmetic on nibbles,
// then scaled by one f16→f32 conversion. This saves ~8x memory bandwidth.

const std = @import("std");

/// Q4_K block-wise dot product: compute dot(quantized_row, x) directly.
///
/// raw_row: pointer to Q4_K blocks for one matrix row
/// x: input f32 vector
/// n: number of elements (must be multiple of 256)
/// Returns: the dot product as f32
pub fn dotQ4K(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const block_size: usize = 256;
    const block_bytes: usize = 144;
    const n_blocks = n / block_size;

    var total: f32 = 0;

    for (0..n_blocks) |block_idx| {
        const block = raw_row[block_idx * block_bytes ..][0..block_bytes];
        const x_block = x[block_idx * block_size ..][0..block_size];

        // Parse block header
        const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
        const dmin_bits = @as(u16, block[2]) | (@as(u16, block[3]) << 8);
        const d = f16ToF32(d_bits);
        const dmin = f16ToF32(dmin_bits);

        // Parse sub-block scales and mins from the 12 scale bytes
        const scale_bytes = block[4..16];

        // 8 sub-blocks of 32 weights each
        // Scales and mins are packed in 4-bit nibbles
        var sub_scales: [8]f32 = undefined;
        var sub_mins: [8]f32 = undefined;

        for (0..8) |sb| {
            const sc: u8 = if (sb < 4)
                scale_bytes[sb] & 0x3F
            else
                scale_bytes[sb - 4 + 4] & 0x3F;
            const mn: u8 = if (sb < 4)
                scale_bytes[sb + 4] & 0x3F
            else
                scale_bytes[sb] & 0x3F;

            sub_scales[sb] = d * @as(f32, @floatFromInt(sc));
            sub_mins[sb] = dmin * @as(f32, @floatFromInt(mn));
        }

        // Data: 128 bytes = 256 nibbles
        const data = block[16..144];

        // Process 8 sub-blocks of 32 weights
        for (0..8) |sb| {
            const sc = sub_scales[sb];
            const mn = sub_mins[sb];
            const x_sub = x_block[sb * 32 ..][0..32];

            var sum_qx: f32 = 0;
            var sum_x: f32 = 0;

            // Each sub-block: 32 weights = 16 bytes of nibbles
            const data_offset = sb * 16;
            for (0..16) |j| {
                const byte = data[data_offset + j];
                const lo: f32 = @floatFromInt(byte & 0xF);
                const hi: f32 = @floatFromInt(byte >> 4);

                sum_qx += lo * x_sub[j * 2];
                sum_qx += hi * x_sub[j * 2 + 1];
                sum_x += x_sub[j * 2] + x_sub[j * 2 + 1];
            }

            total += sc * sum_qx - mn * sum_x;
        }
    }

    return total;
}

/// Q8_0 dot product: simpler format, higher precision.
/// Block: 2 bytes f16 scale + 32 int8 values = 34 bytes per 32 weights.
pub fn dotQ8_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const block_size: usize = 32;
    const block_bytes: usize = 34;
    const n_blocks = n / block_size;

    var total: f32 = 0;

    for (0..n_blocks) |block_idx| {
        const block = raw_row[block_idx * block_bytes ..][0..block_bytes];
        const x_block = x[block_idx * block_size ..][0..block_size];

        const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
        const d = f16ToF32(d_bits);

        var sum: f32 = 0;
        for (0..32) |j| {
            const q: i8 = @bitCast(block[2 + j]);
            sum += @as(f32, @floatFromInt(q)) * x_block[j];
        }

        total += d * sum;
    }

    return total;
}

/// Q4_0 dot product: simplest 4-bit format.
/// Block: 2 bytes f16 scale + 16 bytes data = 18 bytes per 32 weights.
pub fn dotQ4_0(raw_row: []const u8, x: []const f32, n: usize) f32 {
    const block_size: usize = 32;
    const block_bytes: usize = 18;
    const n_blocks = n / block_size;

    var total: f32 = 0;

    for (0..n_blocks) |block_idx| {
        const block = raw_row[block_idx * block_bytes ..][0..block_bytes];
        const x_block = x[block_idx * block_size ..][0..block_size];

        const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
        const d = f16ToF32(d_bits);

        var sum: f32 = 0;
        for (0..16) |j| {
            const byte = block[2 + j];
            const lo = @as(f32, @floatFromInt(byte & 0xF)) - 8.0;
            const hi = @as(f32, @floatFromInt(byte >> 4)) - 8.0;
            sum += lo * x_block[j * 2];
            sum += hi * x_block[j * 2 + 1];
        }

        total += d * sum;
    }

    return total;
}

/// Matrix-vector multiply with quantized weights.
/// out[i] = dot(W_quantized[i], x) for each output dimension.
///
/// raw_weights: raw quantized data for entire weight matrix
/// x: input f32 vector (length = n_cols)
/// out: output f32 vector (length = n_rows)
/// n_cols: input dimension (columns)
/// n_rows: output dimension (rows)
/// tensor_type: quantization format
pub fn matmulQuantized(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
    tensor_type: u32,
) void {
    // Bytes per row = ceil(n_cols / block_size) * block_bytes
    const row_bytes = switch (tensor_type) {
        2 => (n_cols / 32) * 18, // Q4_0
        8 => (n_cols / 32) * 34, // Q8_0
        12 => (n_cols / 256) * 144, // Q4_K
        else => n_cols * 4, // F32 fallback
    };

    for (0..n_rows) |i| {
        const row_data = raw_weights[i * row_bytes ..][0..row_bytes];
        out[i] = switch (tensor_type) {
            2 => dotQ4_0(row_data, x[0..n_cols], n_cols),
            8 => dotQ8_0(row_data, x[0..n_cols], n_cols),
            12 => dotQ4K(row_data, x[0..n_cols], n_cols),
            else => blk: {
                // F32 fallback: interpret as f32 array
                var sum: f32 = 0;
                const f32_row: [*]const f32 = @ptrCast(@alignCast(row_data.ptr));
                for (0..n_cols) |j| sum += f32_row[j] * x[j];
                break :blk sum;
            },
        };
    }
}

/// Convert f16 bits to f32.
fn f16ToF32(bits: u16) f32 {
    const f16_val: f16 = @bitCast(bits);
    return @floatCast(f16_val);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "f16 conversion" {
    // f16 1.0 = 0x3C00
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f16ToF32(0x3C00), 0.01);
    // f16 0.5 = 0x3800
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), f16ToF32(0x3800), 0.01);
}

test "Q4_0 dot product" {
    // Create a simple Q4_0 block: scale=1.0, all nibbles=8 (centered at 0)
    var block: [18]u8 = undefined;
    // f16 1.0 = 0x3C00
    block[0] = 0x00;
    block[1] = 0x3C;
    // All nibbles = 8 → value = 8-8 = 0
    @memset(block[2..18], 0x88);

    const x = [_]f32{1.0} ** 32;
    const result = dotQ4_0(&block, &x, 32);
    // All weights are 0 after centering, so dot product should be 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.1);
}

test "Q8_0 dot product" {
    // Create Q8_0 block: scale=0.5, values=[1,1,1,...,1]
    var block: [34]u8 = undefined;
    // f16 0.5 = 0x3800
    block[0] = 0x00;
    block[1] = 0x38;
    // All int8 values = 1
    @memset(block[2..34], 1);

    const x = [_]f32{2.0} ** 32;
    const result = dotQ8_0(&block, &x, 32);
    // Expected: 0.5 * sum(1 * 2.0) = 0.5 * 64 = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.1);
}

test "matmulQuantized with Q4_0" {
    // 2 rows × 32 cols, Q4_0
    var weights: [36]u8 = undefined; // 2 × 18 bytes
    // Row 0: scale=1, nibbles=9 (value = 9-8 = 1)
    weights[0] = 0x00;
    weights[1] = 0x3C; // f16 1.0
    @memset(weights[2..18], 0x99); // nibble 9 = value 1
    // Row 1: scale=1, nibbles=7 (value = 7-8 = -1)
    weights[18] = 0x00;
    weights[19] = 0x3C;
    @memset(weights[20..36], 0x77); // nibble 7 = value -1

    const x = [_]f32{1.0} ** 32;
    var out: [2]f32 = undefined;
    matmulQuantized(&weights, &x, &out, 32, 2, 2); // type 2 = Q4_0

    // Row 0: all weights ≈ 1, dot with all 1s ≈ 32
    try std.testing.expect(out[0] > 20);
    // Row 1: all weights ≈ -1, dot with all 1s ≈ -32
    try std.testing.expect(out[1] < -20);
}
