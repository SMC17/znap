// SIMD-accelerated vector operations for kaigr-tensor.
//
// Uses Zig's @Vector built-in which compiles to native SIMD instructions
// (SSE/AVX on x86, NEON on ARM). All operations work on comptime-known
// lengths, so the compiler generates optimal unrolled code for each shape.
//
// These are the inner-loop primitives. Every nanosecond matters here.

const std = @import("std");

// ── Core vector operations ──────────────────────────────────────

/// SIMD dot product of two fixed-length vectors.
/// This is THE hot path — called for every weight×feature multiplication.
pub fn vdot(comptime N: usize, a: *const [N]f32, b: *const [N]f32) f32 {
    // Use SIMD vectors for the bulk, scalar for remainder
    const simd_width = simdWidth();

    if (N >= simd_width) {
        const full_iters = N / simd_width;
        const remainder = N % simd_width;

        var acc: @Vector(simd_width, f32) = @splat(0.0);

        for (0..full_iters) |i| {
            const offset = i * simd_width;
            const va: @Vector(simd_width, f32) = a[offset..][0..simd_width].*;
            const vb: @Vector(simd_width, f32) = b[offset..][0..simd_width].*;
            acc += va * vb;
        }

        var result = @reduce(.Add, acc);

        // Handle remainder
        if (remainder > 0) {
            const offset = full_iters * simd_width;
            for (0..remainder) |i| {
                result += a[offset + i] * b[offset + i];
            }
        }

        return result;
    } else {
        // Vectors smaller than SIMD width — scalar fallback
        var result: f32 = 0.0;
        for (0..N) |i| {
            result += a[i] * b[i];
        }
        return result;
    }
}

/// Element-wise add.
pub fn vadd(comptime N: usize, a: *const [N]f32, b: *const [N]f32) [N]f32 {
    return vecBinOp(N, a, b, .add);
}

/// Element-wise subtract.
pub fn vsub(comptime N: usize, a: *const [N]f32, b: *const [N]f32) [N]f32 {
    return vecBinOp(N, a, b, .sub);
}

/// Element-wise multiply.
pub fn vmul(comptime N: usize, a: *const [N]f32, b: *const [N]f32) [N]f32 {
    return vecBinOp(N, a, b, .mul);
}

/// Scalar multiply.
pub fn vscale(comptime N: usize, a: *const [N]f32, scalar: f32) [N]f32 {
    const simd_width = simdWidth();
    var result: [N]f32 = undefined;

    if (N >= simd_width) {
        const full_iters = N / simd_width;
        const remainder = N % simd_width;
        const vs: @Vector(simd_width, f32) = @splat(scalar);

        for (0..full_iters) |i| {
            const offset = i * simd_width;
            const va: @Vector(simd_width, f32) = a[offset..][0..simd_width].*;
            const vr = va * vs;
            result[offset..][0..simd_width].* = vr;
        }

        if (remainder > 0) {
            const offset = full_iters * simd_width;
            for (0..remainder) |i| {
                result[offset + i] = a[offset + i] * scalar;
            }
        }
    } else {
        for (0..N) |i| {
            result[i] = a[i] * scalar;
        }
    }

    return result;
}

/// Sum all elements.
pub fn vsum(comptime N: usize, a: *const [N]f32) f32 {
    const simd_width = simdWidth();

    if (N >= simd_width) {
        const full_iters = N / simd_width;
        const remainder = N % simd_width;

        var acc: @Vector(simd_width, f32) = @splat(0.0);
        for (0..full_iters) |i| {
            const offset = i * simd_width;
            const va: @Vector(simd_width, f32) = a[offset..][0..simd_width].*;
            acc += va;
        }

        var result = @reduce(.Add, acc);
        if (remainder > 0) {
            const offset = full_iters * simd_width;
            for (0..remainder) |i| {
                result += a[offset + i];
            }
        }
        return result;
    } else {
        var result: f32 = 0.0;
        for (0..N) |i| {
            result += a[i];
        }
        return result;
    }
}

/// Max element.
pub fn vmax(comptime N: usize, a: *const [N]f32) f32 {
    var m: f32 = a[0];
    for (1..N) |i| {
        if (a[i] > m) m = a[i];
    }
    return m;
}

/// Argmax — index of the maximum element.
pub fn vargmax(comptime N: usize, a: *const [N]f32) usize {
    var best: usize = 0;
    var best_val: f32 = a[0];
    for (1..N) |i| {
        if (a[i] > best_val) {
            best_val = a[i];
            best = i;
        }
    }
    return best;
}

/// ReLU activation: max(0, x).
pub fn vrelu(comptime N: usize, a: *const [N]f32) [N]f32 {
    const simd_width = simdWidth();
    var result: [N]f32 = undefined;

    if (N >= simd_width) {
        const full_iters = N / simd_width;
        const remainder = N % simd_width;
        const vzero: @Vector(simd_width, f32) = @splat(0.0);

        for (0..full_iters) |i| {
            const offset = i * simd_width;
            const va: @Vector(simd_width, f32) = a[offset..][0..simd_width].*;
            result[offset..][0..simd_width].* = @max(va, vzero);
        }

        if (remainder > 0) {
            const offset = full_iters * simd_width;
            for (0..remainder) |i| {
                result[offset + i] = @max(a[offset + i], 0.0);
            }
        }
    } else {
        for (0..N) |i| {
            result[i] = @max(a[i], 0.0);
        }
    }

    return result;
}

/// Softmax: exp(x_i) / sum(exp(x_j)) with numerical stability (subtract max first).
pub fn vsoftmax(comptime N: usize, a: *const [N]f32) [N]f32 {
    // Subtract max for numerical stability
    const max_val = vmax(N, a);

    var result: [N]f32 = undefined;
    var sum_exp: f32 = 0.0;

    for (0..N) |i| {
        result[i] = @exp(a[i] - max_val);
        sum_exp += result[i];
    }

    // Normalize
    const inv_sum = 1.0 / sum_exp;
    for (0..N) |i| {
        result[i] *= inv_sum;
    }

    return result;
}

// ── Internal helpers ─────────────────────────────────────────────

const BinOp = enum { add, sub, mul };

fn vecBinOp(comptime N: usize, a: *const [N]f32, b: *const [N]f32, comptime op: BinOp) [N]f32 {
    const simd_width = simdWidth();
    var result: [N]f32 = undefined;

    if (N >= simd_width) {
        const full_iters = N / simd_width;
        const remainder = N % simd_width;

        for (0..full_iters) |i| {
            const offset = i * simd_width;
            const va: @Vector(simd_width, f32) = a[offset..][0..simd_width].*;
            const vb: @Vector(simd_width, f32) = b[offset..][0..simd_width].*;
            const vr = switch (op) {
                .add => va + vb,
                .sub => va - vb,
                .mul => va * vb,
            };
            result[offset..][0..simd_width].* = vr;
        }

        if (remainder > 0) {
            const offset = full_iters * simd_width;
            for (0..remainder) |i| {
                result[offset + i] = switch (op) {
                    .add => a[offset + i] + b[offset + i],
                    .sub => a[offset + i] - b[offset + i],
                    .mul => a[offset + i] * b[offset + i],
                };
            }
        }
    } else {
        for (0..N) |i| {
            result[i] = switch (op) {
                .add => a[i] + b[i],
                .sub => a[i] - b[i],
                .mul => a[i] * b[i],
            };
        }
    }

    return result;
}

/// Detect the best SIMD width for the current target.
/// Returns comptime int for use with @Vector.
fn simdWidth() comptime_int {
    // ARM NEON: 4 × f32, x86 SSE: 4 × f32, AVX: 8 × f32
    return switch (@import("builtin").cpu.arch) {
        .x86_64 => 8, // AVX
        .aarch64 => 4, // NEON
        else => 4,
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "vdot basic" {
    const a = [4]f32{ 1, 2, 3, 4 };
    const b = [4]f32{ 1, 1, 1, 1 };
    try std.testing.expectEqual(@as(f32, 10.0), vdot(4, &a, &b));
}

test "vdot 30-dim (KAiGR feature vector)" {
    var a: [30]f32 = undefined;
    var b: [30]f32 = undefined;
    for (0..30) |i| {
        a[i] = 1.0;
        b[i] = @as(f32, @floatFromInt(i));
    }
    // sum of 0..29 = 29*30/2 = 435
    try std.testing.expectEqual(@as(f32, 435.0), vdot(30, &a, &b));
}

test "vadd" {
    const a = [3]f32{ 1, 2, 3 };
    const b = [3]f32{ 10, 20, 30 };
    const c = vadd(3, &a, &b);
    try std.testing.expectEqual(@as(f32, 11.0), c[0]);
    try std.testing.expectEqual(@as(f32, 33.0), c[2]);
}

test "vrelu" {
    const a = [4]f32{ -1, 0, 1, -0.5 };
    const r = vrelu(4, &a);
    try std.testing.expectEqual(@as(f32, 0.0), r[0]);
    try std.testing.expectEqual(@as(f32, 0.0), r[3]);
    try std.testing.expectEqual(@as(f32, 1.0), r[2]);
}

test "vsoftmax sums to 1" {
    const a = [4]f32{ 1, 2, 3, 4 };
    const s = vsoftmax(4, &a);
    var total: f32 = 0;
    for (s) |v| total += v;
    try std.testing.expect(@abs(total - 1.0) < 1e-5);
}

test "vsoftmax preserves order" {
    const a = [3]f32{ 1, 3, 2 };
    const s = vsoftmax(3, &a);
    try std.testing.expect(s[1] > s[2]); // 3 > 2
    try std.testing.expect(s[2] > s[0]); // 2 > 1
}

test "vargmax" {
    const a = [5]f32{ 0.1, 0.9, 0.3, 0.7, 0.2 };
    try std.testing.expectEqual(@as(usize, 1), vargmax(5, &a));
}
