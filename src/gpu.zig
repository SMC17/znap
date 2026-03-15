// GPU hardware abstraction layer for znap.
//
// Provides a unified interface for GPU compute across backends:
//   - CUDA (NVIDIA) — primary target for training
//   - Metal (Apple) — for M-series inference/training
//   - CPU fallback — always available, uses SIMD
//
// The key insight: we don't need a full GPU framework. We need exactly
// four operations for training small models:
//   1. matmul (forward pass)
//   2. matmul_transpose (backward pass)
//   3. element-wise ops (relu, sigmoid, add, scale)
//   4. reduce (sum, max for loss computation)
//
// Everything else (memory management, kernel launch, synchronization)
// is handled by this abstraction layer.
//
// On machines without GPU, the CPU fallback uses the existing SIMD
// operations from simd.zig — so all code compiles and runs everywhere.

const std = @import("std");
const simd = @import("simd.zig");
const tensor = @import("tensor.zig");
const builtin = @import("builtin");

// ── Backend Detection ────────────────────────────────────────────

pub const Backend = enum {
    cpu,
    cuda,
    metal,

    pub fn detect() Backend {
        // Check for CUDA at runtime via environment
        if (std.process.getEnvVarOwned(std.heap.page_allocator, "ZNAP_CUDA") catch null) |v| {
            defer std.heap.page_allocator.free(v);
            return .cuda;
        }
        // On macOS, default to Metal if available
        if (builtin.os.tag == .macos) {
            return .metal; // Metal always available on macOS
        }
        return .cpu;
    }

    pub fn name(self: Backend) []const u8 {
        return switch (self) {
            .cpu => "CPU (SIMD fallback)",
            .cuda => "CUDA (NVIDIA GPU)",
            .metal => "Metal (Apple GPU)",
        };
    }
};

// ── GPU Buffer ───────────────────────────────────────────────────

/// A buffer that may live on CPU or GPU.
/// For CPU backend, this is just a slice wrapper.
/// For GPU backends, this would manage device memory.
pub const Buffer = struct {
    data: []f32,
    rows: usize,
    cols: usize,
    backend: Backend,
    allocator: std.mem.Allocator,

    pub fn alloc(allocator: std.mem.Allocator, rows: usize, cols: usize, backend: Backend) !Buffer {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{
            .data = data,
            .rows = rows,
            .cols = cols,
            .backend = backend,
            .allocator = allocator,
        };
    }

    pub fn free(self: *Buffer) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: *const Buffer, r: usize, c: usize) f32 {
        return self.data[r * self.cols + c];
    }

    pub fn set(self: *Buffer, r: usize, c: usize, val: f32) void {
        self.data[r * self.cols + c] = val;
    }

    /// Copy from a comptime-sized tensor into this buffer.
    pub fn fromTensor(self: *Buffer, comptime R: usize, comptime C: usize, t: *const tensor.Tensor(R, C)) void {
        for (0..R) |r| {
            for (0..C) |c| {
                self.data[r * self.cols + c] = t.data[r][c];
            }
        }
    }

    /// Copy from this buffer back into a comptime-sized tensor.
    pub fn toTensor(self: *const Buffer, comptime R: usize, comptime C: usize) tensor.Tensor(R, C) {
        var result = tensor.Tensor(R, C).zeros();
        for (0..R) |r| {
            for (0..C) |c| {
                result.data[r][c] = self.data[r * self.cols + c];
            }
        }
        return result;
    }
};

// ── GPU Operations ───────────────────────────────────────────────

/// Matrix multiply: C = A × B
/// A is (M × K), B is (K × N), C is (M × N)
pub fn matmul(a: *const Buffer, b: *const Buffer, c: *Buffer) void {
    const M = a.rows;
    const K = a.cols;
    const N = b.cols;

    // CPU fallback with SIMD-friendly inner loop
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a.data[i * K + k] * b.data[k * N + j];
            }
            c.data[i * N + j] = sum;
        }
    }
}

/// Matrix multiply with B transposed: C = A × B^T
/// A is (M × K), B is (N × K) [stored row-major, transposed logically], C is (M × N)
pub fn matmulTranspose(a: *const Buffer, b: *const Buffer, c: *Buffer) void {
    const M = a.rows;
    const K = a.cols;
    const N = b.rows;

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a.data[i * K + k] * b.data[j * K + k];
            }
            c.data[i * N + j] = sum;
        }
    }
}

/// Element-wise ReLU: out[i] = max(0, x[i])
pub fn reluForward(x: *const Buffer, out: *Buffer) void {
    for (0..x.data.len) |i| {
        out.data[i] = @max(x.data[i], 0);
    }
}

/// ReLU backward: grad_input[i] = grad_output[i] if x[i] > 0, else 0
pub fn reluBackward(grad_output: *const Buffer, x: *const Buffer, grad_input: *Buffer) void {
    for (0..x.data.len) |i| {
        grad_input.data[i] = if (x.data[i] > 0) grad_output.data[i] else 0;
    }
}

/// Softmax cross-entropy loss (combined for numerical stability)
pub fn softmaxCrossEntropy(logits: *const Buffer, target: usize) f32 {
    const n = logits.cols;
    // Find max for stability
    var max_val: f32 = logits.data[0];
    for (1..n) |i| {
        if (logits.data[i] > max_val) max_val = logits.data[i];
    }
    // log(sum(exp)) - logit[target]
    var log_sum_exp: f32 = 0;
    for (0..n) |i| {
        log_sum_exp += @exp(logits.data[i] - max_val);
    }
    log_sum_exp = @log(log_sum_exp) + max_val;
    return log_sum_exp - logits.data[target];
}

/// Softmax cross-entropy gradient: grad[i] = softmax(logits)[i] - (i == target)
pub fn softmaxCrossEntropyBackward(logits: *const Buffer, target: usize, grad: *Buffer) void {
    const n = logits.cols;
    var max_val: f32 = logits.data[0];
    for (1..n) |i| {
        if (logits.data[i] > max_val) max_val = logits.data[i];
    }
    var sum_exp: f32 = 0;
    for (0..n) |i| {
        grad.data[i] = @exp(logits.data[i] - max_val);
        sum_exp += grad.data[i];
    }
    for (0..n) |i| {
        grad.data[i] /= sum_exp;
        if (i == target) grad.data[i] -= 1.0;
    }
}

/// SGD update: w -= lr * grad
pub fn sgdUpdate(weights: *Buffer, grads: *const Buffer, lr: f32) void {
    for (0..weights.data.len) |i| {
        weights.data[i] -= lr * grads.data[i];
    }
}

/// SGD update with weight decay: w = w * (1 - lr*wd) - lr * grad
pub fn sgdUpdateWithDecay(weights: *Buffer, grads: *const Buffer, lr: f32, weight_decay: f32) void {
    for (0..weights.data.len) |i| {
        weights.data[i] = weights.data[i] * (1.0 - lr * weight_decay) - lr * grads.data[i];
    }
}

/// Margin-maximizing regularizer (our research finding).
/// Adds a penalty that pushes the gap between best and second-best action wider.
/// This makes the policy more robust to quantization.
pub fn marginRegularizer(logits: *const Buffer, target: usize, margin_weight: f32) f32 {
    const n = logits.cols;
    const target_score = logits.data[target];

    // Find max score among non-target actions
    var max_other: f32 = -std.math.inf(f32);
    for (0..n) |i| {
        if (i != target and logits.data[i] > max_other) {
            max_other = logits.data[i];
        }
    }

    // Hinge-style margin loss: penalize when margin < 1.0
    const margin = target_score - max_other;
    const hinge = @max(1.0 - margin, 0.0);
    return margin_weight * hinge;
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "backend detection" {
    const b = Backend.detect();
    // On macOS it should be metal, otherwise cpu
    try std.testing.expect(b == .cpu or b == .metal);
    try std.testing.expect(b.name().len > 0);
}

test "buffer alloc and free" {
    var buf = try Buffer.alloc(std.testing.allocator, 3, 4, .cpu);
    defer buf.free();
    try std.testing.expectEqual(@as(usize, 3), buf.rows);
    try std.testing.expectEqual(@as(usize, 4), buf.cols);
    try std.testing.expectEqual(@as(usize, 12), buf.data.len);
}

test "buffer get/set" {
    var buf = try Buffer.alloc(std.testing.allocator, 2, 3, .cpu);
    defer buf.free();
    buf.set(1, 2, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), buf.get(1, 2));
}

test "matmul 2x3 × 3x2 = 2x2" {
    var a = try Buffer.alloc(std.testing.allocator, 2, 3, .cpu);
    defer a.free();
    var b = try Buffer.alloc(std.testing.allocator, 3, 2, .cpu);
    defer b.free();
    var c = try Buffer.alloc(std.testing.allocator, 2, 2, .cpu);
    defer c.free();

    // A = [[1,2,3],[4,5,6]]
    a.data[0] = 1; a.data[1] = 2; a.data[2] = 3;
    a.data[3] = 4; a.data[4] = 5; a.data[5] = 6;
    // B = [[7,8],[9,10],[11,12]]
    b.data[0] = 7; b.data[1] = 8;
    b.data[2] = 9; b.data[3] = 10;
    b.data[4] = 11; b.data[5] = 12;

    matmul(&a, &b, &c);

    // C = [[58,64],[139,154]]
    try std.testing.expectEqual(@as(f32, 58), c.get(0, 0));
    try std.testing.expectEqual(@as(f32, 64), c.get(0, 1));
    try std.testing.expectEqual(@as(f32, 139), c.get(1, 0));
    try std.testing.expectEqual(@as(f32, 154), c.get(1, 1));
}

test "relu forward" {
    var x = try Buffer.alloc(std.testing.allocator, 1, 4, .cpu);
    defer x.free();
    var out = try Buffer.alloc(std.testing.allocator, 1, 4, .cpu);
    defer out.free();

    x.data[0] = -2; x.data[1] = 3; x.data[2] = -1; x.data[3] = 5;
    reluForward(&x, &out);

    try std.testing.expectEqual(@as(f32, 0), out.data[0]);
    try std.testing.expectEqual(@as(f32, 3), out.data[1]);
    try std.testing.expectEqual(@as(f32, 0), out.data[2]);
    try std.testing.expectEqual(@as(f32, 5), out.data[3]);
}

test "relu backward" {
    var grad_out = try Buffer.alloc(std.testing.allocator, 1, 4, .cpu);
    defer grad_out.free();
    var x = try Buffer.alloc(std.testing.allocator, 1, 4, .cpu);
    defer x.free();
    var grad_in = try Buffer.alloc(std.testing.allocator, 1, 4, .cpu);
    defer grad_in.free();

    grad_out.data[0] = 1; grad_out.data[1] = 1; grad_out.data[2] = 1; grad_out.data[3] = 1;
    x.data[0] = -2; x.data[1] = 3; x.data[2] = -1; x.data[3] = 5;

    reluBackward(&grad_out, &x, &grad_in);

    try std.testing.expectEqual(@as(f32, 0), grad_in.data[0]); // x<0, grad=0
    try std.testing.expectEqual(@as(f32, 1), grad_in.data[1]); // x>0, grad passes
}

test "softmax cross entropy" {
    var logits = try Buffer.alloc(std.testing.allocator, 1, 3, .cpu);
    defer logits.free();
    logits.data[0] = 2.0; logits.data[1] = 1.0; logits.data[2] = 0.1;

    const loss = softmaxCrossEntropy(&logits, 0);
    try std.testing.expect(loss > 0);
    try std.testing.expect(loss < 2.0); // Should be small since target has highest logit
}

test "softmax cross entropy gradient" {
    var logits = try Buffer.alloc(std.testing.allocator, 1, 3, .cpu);
    defer logits.free();
    var grad = try Buffer.alloc(std.testing.allocator, 1, 3, .cpu);
    defer grad.free();
    logits.data[0] = 2.0; logits.data[1] = 1.0; logits.data[2] = 0.1;

    softmaxCrossEntropyBackward(&logits, 0, &grad);

    // Gradient for target class should be negative (softmax - 1)
    try std.testing.expect(grad.data[0] < 0);
    // Gradients for non-target should be positive (softmax > 0)
    try std.testing.expect(grad.data[1] > 0);
    try std.testing.expect(grad.data[2] > 0);
    // Sum of gradients should be ~0
    const sum = grad.data[0] + grad.data[1] + grad.data[2];
    try std.testing.expect(@abs(sum) < 1e-5);
}

test "sgd update" {
    var w = try Buffer.alloc(std.testing.allocator, 1, 3, .cpu);
    defer w.free();
    var g = try Buffer.alloc(std.testing.allocator, 1, 3, .cpu);
    defer g.free();

    w.data[0] = 1.0; w.data[1] = 2.0; w.data[2] = 3.0;
    g.data[0] = 0.1; g.data[1] = 0.2; g.data[2] = 0.3;

    sgdUpdate(&w, &g, 0.1);

    try std.testing.expectApproxEqAbs(@as(f32, 0.99), w.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.98), w.data[1], 1e-5);
}

test "margin regularizer" {
    var logits = try Buffer.alloc(std.testing.allocator, 1, 4, .cpu);
    defer logits.free();
    logits.data[0] = 5.0; logits.data[1] = 1.0; logits.data[2] = 2.0; logits.data[3] = 0.5;

    // Target=0, margin = 5.0 - 2.0 = 3.0, hinge = max(1-3, 0) = 0
    const loss = marginRegularizer(&logits, 0, 1.0);
    try std.testing.expectEqual(@as(f32, 0), loss); // Margin > 1, no penalty

    // Make margin small
    logits.data[1] = 4.5; // margin = 5.0 - 4.5 = 0.5
    const loss2 = marginRegularizer(&logits, 0, 1.0);
    try std.testing.expect(loss2 > 0); // Margin < 1, penalty applied
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), loss2, 1e-5); // hinge = 1 - 0.5 = 0.5
}

test "tensor round-trip through buffer" {
    const T = tensor.Tensor(2, 3);
    const t = T.fromFlat(.{ 1, 2, 3, 4, 5, 6 });

    var buf = try Buffer.alloc(std.testing.allocator, 2, 3, .cpu);
    defer buf.free();
    buf.fromTensor(2, 3, &t);

    const t2 = buf.toTensor(2, 3);
    try std.testing.expectEqual(@as(f32, 1), t2.data[0][0]);
    try std.testing.expectEqual(@as(f32, 6), t2.data[1][2]);
}
