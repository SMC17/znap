// Fixed-shape tensor type with comptime dimensions.
//
// Zero allocations. All shapes known at compile time. SIMD-accelerated.
// The compiler generates specialized code for each shape — no runtime
// dispatch, no dynamic memory, no bounds checking in release builds.

const std = @import("std");
const simd = @import("simd.zig");

/// A fixed-shape 2D tensor. Rows x Cols known at compile time.
/// Data is stored in row-major order on the stack.
pub fn Tensor(comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        data: [Rows][Cols]f32,

        const Self = @This();
        pub const rows = Rows;
        pub const cols = Cols;
        pub const numel = Rows * Cols;

        pub fn zeros() Self {
            return .{ .data = [_][Cols]f32{[_]f32{0.0} ** Cols} ** Rows };
        }

        pub fn fill(val: f32) Self {
            return .{ .data = [_][Cols]f32{[_]f32{val} ** Cols} ** Rows };
        }

        pub fn fromFlat(values: [Rows * Cols]f32) Self {
            var result: Self = undefined;
            for (0..Rows) |r| {
                for (0..Cols) |c| {
                    result.data[r][c] = values[r * Cols + c];
                }
            }
            return result;
        }

        pub fn fromArray(arr: [Rows][Cols]f32) Self {
            return .{ .data = arr };
        }

        pub fn get(self: *const Self, r: usize, c: usize) f32 {
            return self.data[r][c];
        }

        pub fn set(self: *Self, r: usize, c: usize, val: f32) void {
            self.data[r][c] = val;
        }

        pub fn row(self: *const Self, r: usize) *const [Cols]f32 {
            return &self.data[r];
        }

        pub fn rowMut(self: *Self, r: usize) *[Cols]f32 {
            return &self.data[r];
        }

        // ── Element-wise ops ─────────────────────────────────────

        pub fn add(self: *const Self, other: *const Self) Self {
            var result: Self = undefined;
            for (0..Rows) |r| {
                result.data[r] = simd.vadd(Cols, &self.data[r], &other.data[r]);
            }
            return result;
        }

        pub fn sub(self: *const Self, other: *const Self) Self {
            var result: Self = undefined;
            for (0..Rows) |r| {
                result.data[r] = simd.vsub(Cols, &self.data[r], &other.data[r]);
            }
            return result;
        }

        pub fn scale(self: *const Self, scalar: f32) Self {
            var result: Self = undefined;
            for (0..Rows) |r| {
                result.data[r] = simd.vscale(Cols, &self.data[r], scalar);
            }
            return result;
        }

        pub fn hadamard(self: *const Self, other: *const Self) Self {
            var result: Self = undefined;
            for (0..Rows) |r| {
                result.data[r] = simd.vmul(Cols, &self.data[r], &other.data[r]);
            }
            return result;
        }

        // ── Reductions ───────────────────────────────────────────

        pub fn sum(self: *const Self) f32 {
            var total: f32 = 0.0;
            for (0..Rows) |r| {
                total += simd.vsum(Cols, &self.data[r]);
            }
            return total;
        }

        pub fn max(self: *const Self) f32 {
            var m: f32 = -std.math.inf(f32);
            for (0..Rows) |r| {
                const row_max = simd.vmax(Cols, &self.data[r]);
                if (row_max > m) m = row_max;
            }
            return m;
        }

        pub fn argmax(self: *const Self) usize {
            return simd.vargmax(Cols, &self.data[0]);
        }

        // ── Activations ──────────────────────────────────────────

        pub fn relu(self: *Self) void {
            for (0..Rows) |r| {
                self.data[r] = simd.vrelu(Cols, &self.data[r]);
            }
        }

        pub fn softmax(self: *Self) void {
            for (0..Rows) |r| {
                self.data[r] = simd.vsoftmax(Cols, &self.data[r]);
            }
        }

        pub fn clamp(self: *Self, lo: f32, hi: f32) void {
            for (0..Rows) |r| {
                for (0..Cols) |c| {
                    if (self.data[r][c] < lo) self.data[r][c] = lo;
                    if (self.data[r][c] > hi) self.data[r][c] = hi;
                }
            }
        }
    };
}

/// Comptime matrix-vector multiply: (M x K) weights x K features -> M scores.
/// This is the hot path for policy evaluation.
pub fn matvec(
    comptime M: usize,
    comptime K: usize,
    weights: *const Tensor(M, K),
    vec: *const [K]f32,
) [M]f32 {
    var result: [M]f32 = [_]f32{0.0} ** M;
    for (0..M) |i| {
        result[i] = simd.vdot(K, &weights.data[i], vec);
    }
    return result;
}

/// Comptime matrix multiply: (M x K) x (K x N) -> (M x N).
pub fn matmul(
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    a: *const Tensor(M, K),
    b: *const Tensor(K, N),
) Tensor(M, N) {
    var result = Tensor(M, N).zeros();
    for (0..M) |i| {
        for (0..K) |k| {
            const a_ik = a.data[i][k];
            for (0..N) |j| {
                result.data[i][j] += a_ik * b.data[k][j];
            }
        }
    }
    return result;
}

/// 1D vector alias.
pub fn Vec(comptime N: usize) type {
    return Tensor(1, N);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "zeros" {
    const t = Tensor(3, 4).zeros();
    try std.testing.expectEqual(@as(f32, 0.0), t.data[0][0]);
    try std.testing.expectEqual(@as(f32, 0.0), t.data[2][3]);
}

test "fill" {
    const t = Tensor(2, 3).fill(1.5);
    try std.testing.expectEqual(@as(f32, 1.5), t.data[1][2]);
}

test "fromFlat" {
    const t = Tensor(2, 3).fromFlat(.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(@as(f32, 1.0), t.data[0][0]);
    try std.testing.expectEqual(@as(f32, 6.0), t.data[1][2]);
}

test "add" {
    const a = Tensor(1, 4).fromFlat(.{ 1, 2, 3, 4 });
    const b = Tensor(1, 4).fromFlat(.{ 10, 20, 30, 40 });
    const c = a.add(&b);
    try std.testing.expectEqual(@as(f32, 11.0), c.data[0][0]);
    try std.testing.expectEqual(@as(f32, 44.0), c.data[0][3]);
}

test "scale" {
    const a = Tensor(1, 3).fromFlat(.{ 2, 4, 6 });
    const b = a.scale(0.5);
    try std.testing.expectEqual(@as(f32, 1.0), b.data[0][0]);
}

test "sum" {
    const t = Tensor(2, 2).fromFlat(.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(f32, 10.0), t.sum());
}

test "argmax" {
    const t = Tensor(1, 5).fromFlat(.{ 0.1, 0.9, 0.3, 0.7, 0.2 });
    try std.testing.expectEqual(@as(usize, 1), t.argmax());
}

test "relu" {
    var t = Tensor(1, 4).fromFlat(.{ -2, 3, -1, 5 });
    t.relu();
    try std.testing.expectEqual(@as(f32, 0.0), t.data[0][0]);
    try std.testing.expectEqual(@as(f32, 3.0), t.data[0][1]);
}

test "softmax sums to 1" {
    var t = Tensor(1, 4).fromFlat(.{ 1, 2, 3, 4 });
    t.softmax();
    try std.testing.expect(@abs(t.sum() - 1.0) < 1e-5);
}

test "matvec" {
    const w = Tensor(2, 3).fromFlat(.{ 1, 0, 0, 0, 0, 1 });
    const v = [3]f32{ 0.5, 0.0, 0.8 };
    const r = matvec(2, 3, &w, &v);
    try std.testing.expectEqual(@as(f32, 0.5), r[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), r[1], 1e-6);
}
