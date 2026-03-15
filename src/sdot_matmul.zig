// SDOT-based quantized matmul with Q8_0 input quantization.
//
// The llama.cpp approach: quantize the f32 input vector to Q8_0 format
// (f16 scale per 32 elements + 32 i8 values), then SDOT between
// Q5_0/Q8_0 weight i8s and Q8_0 input i8s.
//
// Q8_0 quantization provides 11 bits of mantissa via f16 scales,
// far more precise than raw f32→i8 (7 bits). This preserves quality
// while enabling integer-only SDOT accumulation.

const std = @import("std");
const builtin = @import("builtin");

/// Q8_0 block for quantized input: f16 scale + 32 i8 values.
/// Total: 34 bytes per 32 elements.
pub const Q8Block = struct {
    scale: f32, // Stored as f32 for computation (converted from f16 for precision)
    values: [32]i8,
};

/// Quantize f32 vector to Q8_0 format (per-block f16 scale + i8 values).
/// This matches ggml's quantize_row_q8_0 exactly.
pub fn quantizeToQ8_0(x: []const f32, blocks: []Q8Block, n: usize) void {
    const BS = 32;
    var b: usize = 0;
    while (b * BS < n) : (b += 1) {
        const xb = x[b * BS ..][0..BS];

        // Find absmax with SIMD
        var absmax: f32 = 0;
        for (xb) |v| {
            const a = @abs(v);
            if (a > absmax) absmax = a;
        }

        // Scale: convert to f16 and back for ggml-compatible precision
        const d: f32 = absmax / 127.0;
        const d_f16: f16 = @floatCast(d);
        const d_back: f32 = @floatCast(d_f16); // This is what ggml stores
        blocks[b].scale = d_back;

        // Quantize with the f16-rounded scale
        const inv = if (d_back > 0) 1.0 / d_back else 0.0;
        for (0..BS) |j| {
            const v = x[b * BS + j] * inv;
            blocks[b].values[j] = @intFromFloat(std.math.clamp(@round(v), -127.0, 127.0));
        }
    }
}

/// SDOT: 32-element i8×i8 dot product → i32.
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

/// Full row SDOT with inline scale handling — NO per-block function call.
/// Processes n_blocks × 32 weights with per-block f16 scale and per-block Q8_0 input scale.
/// All in one asm invocation.
fn sdotRowQ5(
    w_row: [*]const i8,
    w_raw: [*]const u8, // Original Q5_0/Q8_0 blocks (for scales)
    q8_blocks: [*]const Q8Block,
    n_blocks: usize,
    w_block_bytes: usize,
) f32 {
    if (builtin.cpu.arch != .aarch64) {
        var total: f32 = 0;
        for (0..n_blocks) |b| {
            const block = w_raw[b * w_block_bytes ..];
            const ws: f32 = @floatCast(@as(f16, @bitCast(@as(u16, block[0]) | (@as(u16, block[1]) << 8))));
            const dot = sdotBlock(w_row + b * 32, &q8_blocks[b].values);
            total += ws * q8_blocks[b].scale * @as(f32, @floatFromInt(dot));
        }
        return total;
    }

    // NEON assembly: process all blocks with inline scale handling
    var result: f32 = undefined;
    asm volatile (
        // x0 = w_row (i8 weights), x1 = w_raw (Q5_0 blocks for scales)
        // x2 = q8_blocks, x3 = n_blocks, x4 = w_block_bytes
        \\mov x11, x0
        \\mov x12, x1
        \\mov x13, x2
        \\mov x14, x3
        \\mov x15, x4
        \\movi v30.4s, #0
        // Loop over blocks
        \\1:
        \\cbz x14, 2f
        // Load weight scale from Q5_0 block (f16 at offset 0)
        \\ldr h0, [x12]
        \\fcvt s0, h0
        // Load input scale from Q8Block (f32 at offset 0)
        \\ldr s1, [x13]
        // Multiply scales: combined_scale = w_scale × x_scale
        \\fmul s0, s0, s1
        // Load 32 i8 weights
        \\ldr q2, [x11]
        \\ldr q3, [x11, #16]
        // Load 32 i8 input (at offset 4 in Q8Block — after the f32 scale)
        \\ldr q4, [x13, #4]
        \\ldr q5, [x13, #20]
        // SDOT: 2 instructions, 32 multiply-accumulates
        \\movi v6.4s, #0
        \\sdot v6.4s, v2.16b, v4.16b
        \\sdot v6.4s, v3.16b, v5.16b
        // Reduce 4 i32 lanes → scalar
        \\addv s6, v6.4s
        // Convert i32 → f32
        \\scvtf s6, s6
        // Multiply by combined scale
        \\fmul s6, s6, s0
        // Accumulate
        \\fadd s30, s30, s6
        // Advance pointers
        \\add x11, x11, #32
        \\add x12, x12, x15
        \\add x13, x13, #36
        \\sub x14, x14, #1
        \\b 1b
        \\2:
        \\fmov w8, s30
        : [out] "={w8}" (result),
        : [w] "{x0}" (w_row),
          [raw] "{x1}" (w_raw),
          [q8] "{x2}" (q8_blocks),
          [nb] "{x3}" (n_blocks),
          [bb] "{x4}" (w_block_bytes),
        : "v0","v1","v2","v3","v4","v5","v6","v30","x11","x12","x13","x14","x15","memory"
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

/// Pre-dequantized i8 weight matrix for SDOT: per-row f32 scale + flat i8 data.
pub const SdotWeightMatrix = struct {
    scales: []f32, // [n_rows]
    data: []i8, // [n_rows * n_cols]
    n_rows: usize,
    n_cols: usize,

    /// Dequantize a Q5_0 tensor to i8 weights + scales at load time.
    pub fn fromQ5_0(raw: []const u8, n_rows: usize, n_cols: usize, allocator: std.mem.Allocator) !SdotWeightMatrix {
        const n_blocks_per_row = n_cols / 32;
        const row_bytes = n_blocks_per_row * 22;

        var scales = try allocator.alloc(f32, n_rows);
        var data = try allocator.alloc(i8, n_rows * n_cols);

        for (0..n_rows) |i| {
            const raw_row = raw[i * row_bytes ..];
            var row_scale_sum: f32 = 0;

            for (0..n_blocks_per_row) |b| {
                const block = raw_row[b * 22 ..];
                const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const qh = @as(u32, block[2]) | (@as(u32, block[3]) << 8) |
                    (@as(u32, block[4]) << 16) | (@as(u32, block[5]) << 24);

                // Store per-block scale into a running max for the row
                const abs_d = @abs(d);
                if (abs_d > row_scale_sum) row_scale_sum = abs_d;

                // Dequantize to i8 (keeping block-local scale)
                // We store the SCALED i8 values: w_i8 = round(w_float / row_scale * 127)
                // But for SDOT we need per-BLOCK scale, not per-row.
                // So we store the raw 5-bit values as i8 and keep per-block scales.
                for (0..16) |j| {
                    const byte = block[6 + j];
                    const xh_lo: u8 = @intCast((qh >> @as(u5, @intCast(j))) & 1);
                    const xh_hi: u8 = @intCast((qh >> @as(u5, @intCast(j + 16))) & 1);
                    data[i * n_cols + b * 32 + j] = @intCast(@as(i16, (byte & 0xF) | (xh_lo << 4)) - 16);
                    data[i * n_cols + b * 32 + j + 16] = @intCast(@as(i16, (byte >> 4) | (xh_hi << 4)) - 16);
                }
            }
            scales[i] = row_scale_sum;
        }

        return .{ .scales = scales, .data = data, .n_rows = n_rows, .n_cols = n_cols };
    }

    pub fn deinit(self: *SdotWeightMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.scales);
        allocator.free(self.data);
    }
};

/// Row computation for one output element.
fn computeRowSdot(
    w_row: []const i8,
    raw_row: []const u8,
    q8_blocks: []const Q8Block,
    n_cols: usize,
    w_block_bytes: usize,
) f32 {
    const block_size: usize = switch (w_block_bytes) {
        22, 34 => 32,
        210, 144 => 256,
        else => 32,
    };
    const n_main_blocks = n_cols / block_size;
    const sub_blocks_per = block_size / 32;
    var total: f32 = 0;

    for (0..n_main_blocks) |mb| {
        const block = raw_row[mb * w_block_bytes ..];

        if (w_block_bytes == 210) {
            const d: f32 = @floatCast(@as(f16, @bitCast(@as(u16, block[208]) | (@as(u16, block[209]) << 8))));
            const sc = block[192..208];
            for (0..sub_blocks_per) |sb| {
                const sb_idx = mb * sub_blocks_per + sb;
                const w_ptr: [*]const i8 = @ptrCast(w_row[sb_idx * 32 ..].ptr);
                const dot = sdotBlock(w_ptr, &q8_blocks[sb_idx].values);
                const sub_scale: f32 = @floatFromInt(@as(i8, @bitCast(sc[sb])));
                total += d * sub_scale * q8_blocks[sb_idx].scale * @as(f32, @floatFromInt(dot));
            }
        } else if (w_block_bytes == 144) {
            const d: f32 = @floatCast(@as(f16, @bitCast(@as(u16, block[0]) | (@as(u16, block[1]) << 8))));
            for (0..sub_blocks_per) |sb| {
                const sb_idx = mb * sub_blocks_per + sb;
                const w_ptr: [*]const i8 = @ptrCast(w_row[sb_idx * 32 ..].ptr);
                const dot = sdotBlock(w_ptr, &q8_blocks[sb_idx].values);
                total += d * q8_blocks[sb_idx].scale * @as(f32, @floatFromInt(dot));
            }
        } else {
            const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
            const w_scale: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
            const b32 = mb;
            const w_ptr: [*]const i8 = @ptrCast(w_row[b32 * 32 ..].ptr);
            const dot = sdotBlock(w_ptr, &q8_blocks[b32].values);
            total += w_scale * q8_blocks[b32].scale * @as(f32, @floatFromInt(dot));
        }
    }
    return total;
}

const SdotWork = struct {
    w_data: []const i8,
    w_raw: []const u8,
    q8: []const Q8Block,
    out: []f32,
    n_cols: usize,
    w_bb: usize,
    n_mb: usize,
    start: usize,
    end: usize,
};

fn sdotWorker(w: SdotWork) void {
    for (w.start..w.end) |i| {
        w.out[i] = computeRowSdot(
            w.w_data[i * w.n_cols ..][0..w.n_cols],
            w.w_raw[i * w.n_mb * w.w_bb ..],
            w.q8,
            w.n_cols,
            w.w_bb,
        );
    }
}

/// Pre-dequanted SDOT matmul — multithreaded.
pub fn matmulSdotCached(
    w_data: []const i8,
    w_raw: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
    w_block_bytes: usize,
) void {
    const n_32blocks = n_cols / 32;
    const block_size: usize = switch (w_block_bytes) {
        22, 34 => 32,
        210, 144 => 256,
        else => 32,
    };
    const n_main_blocks = n_cols / block_size;

    // Quantize input to Q8_0 ONCE
    var q8_blocks: [256]Q8Block = undefined;
    quantizeToQ8_0(x[0..n_cols], q8_blocks[0..n_32blocks], n_cols);

    // Only multithread vocab projection (49K+ rows). Layer matmuls (<2K rows)
    // are too small — thread spawn overhead (20μs) exceeds the matmul time.
    if (n_rows >= 8192) {
        const n_threads = @min(std.Thread.getCpuCount() catch 1, 8);
        if (n_threads > 1) {
            const rpt = n_rows / n_threads;
            var threads: [8]std.Thread = undefined;
            var spawned: usize = 0;

            for (0..n_threads) |t| {
                const s = t * rpt;
                const e = if (t == n_threads - 1) n_rows else (t + 1) * rpt;
                threads[t] = std.Thread.spawn(.{}, sdotWorker, .{SdotWork{
                    .w_data = w_data, .w_raw = w_raw,
                    .q8 = q8_blocks[0..n_32blocks], .out = out,
                    .n_cols = n_cols, .w_bb = w_block_bytes,
                    .n_mb = n_main_blocks, .start = s, .end = e,
                }}) catch {
                    sdotWorker(.{ .w_data = w_data, .w_raw = w_raw, .q8 = q8_blocks[0..n_32blocks], .out = out, .n_cols = n_cols, .w_bb = w_block_bytes, .n_mb = n_main_blocks, .start = s, .end = e });
                    continue;
                };
                spawned += 1;
            }
            for (0..spawned) |t| threads[t].join();
            return;
        }
    }

    // Single-threaded fallback
    for (0..n_rows) |i| {
        out[i] = computeRowSdot(
            w_data[i * n_cols ..][0..n_cols],
            w_raw[i * n_main_blocks * w_block_bytes ..],
            q8_blocks[0..n_32blocks],
            n_cols,
            w_block_bytes,
        );
    }
}

/// SDOT matmul with PRE-QUANTIZED input — fast path for Q5_0/Q8_0 (32-element blocks).
/// No switch on block type in the inner loop. Minimal overhead.
pub fn matmulSdotPrequant(
    w_data: []const i8,
    w_raw: []const u8,
    q8_blocks: []const Q8Block,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
    w_block_bytes: usize,
) void {
    // Fast path: Q5_0 (22) or Q8_0 (34) — full row assembly, no per-block function call
    if (w_block_bytes == 22 or w_block_bytes == 34) {
        const n_blocks = n_cols / 32;
        for (0..n_rows) |i| {
            const w_row: [*]const i8 = @ptrCast(w_data[i * n_cols ..].ptr);
            const raw_row: [*]const u8 = w_raw.ptr + i * n_blocks * w_block_bytes;
            const q8_ptr: [*]const Q8Block = q8_blocks.ptr;
            out[i] = sdotRowQ5(w_row, raw_row, q8_ptr, n_blocks, w_block_bytes);
        }
        return;
    }

    // Q6_K (210) or Q4_K (144) — 256-element blocks with sub-block scales
    const block_size: usize = 256;
    const n_main_blocks = n_cols / block_size;

    for (0..n_rows) |i| {
        out[i] = computeRowSdot(
            w_data[i * n_cols ..][0..n_cols],
            w_raw[i * n_main_blocks * w_block_bytes ..],
            q8_blocks,
            n_cols,
            w_block_bytes,
        );
    }
}

/// Q5_0 matmul using SDOT with Q8_0 input quantization.
/// Input quantized to Q8_0 ONCE, then SDOT for every row.
pub fn matmulQ5_0(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    const n_blocks = n_cols / 32;
    const row_bytes = n_blocks * 22;

    // Quantize input to Q8_0 ONCE (f16-precision scales)
    var q8_blocks: [256]Q8Block = undefined;
    quantizeToQ8_0(x[0..n_cols], q8_blocks[0..n_blocks], n_cols);

    for (0..n_rows) |i| {
        const raw_row = raw_weights[i * row_bytes ..];
        var total: f32 = 0;

        for (0..n_blocks) |b| {
            const block = raw_row[b * 22 ..];
            const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
            const w_scale: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

            // Dequantize Q5_0 weights to i8
            var w_i8: [32]i8 = undefined;
            dequantQ5_0ToI8(block.ptr, &w_i8);

            // SDOT: i8 weights × i8 input → i32
            const dot = sdotBlock(&w_i8, &q8_blocks[b].values);

            // Dequantize: w_scale × x_scale × i32_sum
            total += w_scale * q8_blocks[b].scale * @as(f32, @floatFromInt(dot));
        }

        out[i] = total;
    }
}

/// Q8_0 matmul using SDOT — weights already i8, just need Q8_0 input.
pub fn matmulQ8_0(
    raw_weights: []const u8,
    x: []const f32,
    out: []f32,
    n_cols: usize,
    n_rows: usize,
) void {
    const n_blocks = n_cols / 32;
    const row_bytes = n_blocks * 34;

    var q8_blocks: [256]Q8Block = undefined;
    quantizeToQ8_0(x[0..n_cols], q8_blocks[0..n_blocks], n_cols);

    for (0..n_rows) |i| {
        const raw_row = raw_weights[i * row_bytes ..];
        var total: f32 = 0;

        for (0..n_blocks) |b| {
            const block = raw_row[b * 34 ..];
            const d_bits = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
            const w_scale: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

            const w_ptr: [*]const i8 = @ptrCast(block[2..].ptr);
            const dot = sdotBlock(w_ptr, &q8_blocks[b].values);
            total += w_scale * q8_blocks[b].scale * @as(f32, @floatFromInt(dot));
        }

        out[i] = total;
    }
}
