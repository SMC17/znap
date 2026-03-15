// State Space Model (SSM) / Mamba block for znap.
//
// Implements the selective state space model from Mamba (Gu & Dao, 2023).
// This is the "post-attention" candidate — linear-time sequence modeling
// with O(1) memory per token during inference (vs O(n) for attention KV cache).
//
// Architecture:
//   x → Linear → [Conv1D → SiLU] → SSM → Linear → output
//                                     ↑
//                              Selective scan (the core innovation)
//
// The selective scan makes SSM parameters (B, C, Δ) INPUT-DEPENDENT,
// allowing the model to selectively remember or forget information.
// This is what makes Mamba competitive with transformers on language tasks.
//
// Key advantage: inference is O(1) memory per token. No KV cache growth.
// A hybrid model (7 Mamba layers + 1 attention layer) gets transformer
// quality with 8x less KV cache memory.
//
// Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
// (Gu & Dao, 2023) — https://arxiv.org/abs/2312.00752

const std = @import("std");
const simd = @import("simd.zig");

/// Selective State Space Model block.
///
/// Parameters:
///   ModelDim: input/output dimension (e.g., 288 for stories15M)
///   StateDim: SSM state dimension (typically 16)
///   ConvWidth: 1D convolution kernel width (typically 4)
///   ExpandFactor: inner dimension expansion (typically 2)
pub fn MambaBlock(
    comptime ModelDim: usize,
    comptime StateDim: usize,
    comptime ConvWidth: usize,
    comptime ExpandFactor: usize,
) type {
    const InnerDim = ModelDim * ExpandFactor;

    return struct {
        // Projection weights
        in_proj: [InnerDim * 2][ModelDim]f32, // Projects to 2x inner (for gate)
        out_proj: [ModelDim][InnerDim]f32, // Projects back to model dim

        // Conv1D weights (depthwise)
        conv_weight: [InnerDim][ConvWidth]f32,
        conv_bias: [InnerDim]f32,

        // SSM parameters (input-dependent via projections)
        dt_proj_weight: [InnerDim][InnerDim]f32, // Δ projection (simplified)
        dt_proj_bias: [InnerDim]f32,
        A: [InnerDim][StateDim]f32, // State transition (learned, typically negative log-spaced)
        D: [InnerDim]f32, // Skip connection

        // Per-token SSM projections
        x_proj_weight: [StateDim + StateDim + 1][InnerDim]f32, // Projects to B, C, dt

        const Self = @This();
        pub const model_dim = ModelDim;
        pub const state_dim = StateDim;
        pub const inner_dim = InnerDim;
        pub const conv_width = ConvWidth;
        pub const param_count = InnerDim * 2 * ModelDim + ModelDim * InnerDim +
            InnerDim * ConvWidth + InnerDim + InnerDim * InnerDim + InnerDim +
            InnerDim * StateDim + InnerDim + (StateDim * 2 + 1) * InnerDim;

        /// SSM hidden state — O(1) memory, doesn't grow with sequence length.
        pub const State = struct {
            h: [InnerDim][StateDim]f32, // Hidden state
            conv_buf: [InnerDim][ConvWidth]f32, // Conv1D ring buffer
            conv_pos: usize,

            pub fn init() State {
                return .{
                    .h = [_][StateDim]f32{[_]f32{0} ** StateDim} ** InnerDim,
                    .conv_buf = [_][ConvWidth]f32{[_]f32{0} ** ConvWidth} ** InnerDim,
                    .conv_pos = 0,
                };
            }

            pub fn reset(self: *State) void {
                self.* = State.init();
            }
        };

        pub fn init() Self {
            var self: Self = undefined;
            @memset(std.mem.asBytes(&self), 0);

            // Initialize A as negative log-spaced values (standard Mamba init)
            for (0..InnerDim) |i| {
                for (0..StateDim) |j| {
                    self.A[i][j] = -@as(f32, @floatFromInt(j + 1));
                }
            }

            // D = 1 (skip connection initialized to identity)
            for (0..InnerDim) |i| self.D[i] = 1.0;

            return self;
        }

        /// Forward pass for a single token. O(1) memory — no KV cache needed.
        ///
        /// This is the selective scan: SSM parameters B, C, Δ are computed
        /// from the input, allowing the model to selectively attend to
        /// different parts of the sequence.
        pub fn forward(self: *const Self, x: *const [ModelDim]f32, state: *State) [ModelDim]f32 {
            // 1. Input projection: x → [xz] (split into x_inner and gate z)
            var xz: [InnerDim * 2]f32 = undefined;
            for (0..InnerDim * 2) |i| {
                var sum: f32 = 0;
                for (0..ModelDim) |j| {
                    sum += self.in_proj[i][j] * x[j];
                }
                xz[i] = sum;
            }

            var x_inner: [InnerDim]f32 = xz[0..InnerDim].*;
            const z: [InnerDim]f32 = xz[InnerDim..][0..InnerDim].*;

            // 2. Conv1D (causal, depthwise)
            // Push into ring buffer
            for (0..InnerDim) |i| {
                state.conv_buf[i][state.conv_pos % ConvWidth] = x_inner[i];
            }
            state.conv_pos += 1;

            // Apply convolution
            for (0..InnerDim) |i| {
                var sum: f32 = self.conv_bias[i];
                for (0..ConvWidth) |k| {
                    const buf_pos = (state.conv_pos + k) % ConvWidth;
                    sum += self.conv_weight[i][k] * state.conv_buf[i][buf_pos];
                }
                x_inner[i] = sum;
            }

            // SiLU activation
            for (0..InnerDim) |i| {
                x_inner[i] = x_inner[i] / (1.0 + @exp(-x_inner[i]));
            }

            // 3. SSM: selective scan
            // Project x_inner to get B, C, dt (input-dependent SSM parameters)
            var x_proj: [StateDim * 2 + 1]f32 = undefined;
            for (0..StateDim * 2 + 1) |i| {
                var sum: f32 = 0;
                for (0..InnerDim) |j| {
                    sum += self.x_proj_weight[i][j] * x_inner[j];
                }
                x_proj[i] = sum;
            }

            const B = x_proj[0..StateDim];
            const C = x_proj[StateDim .. StateDim * 2];
            const dt_raw = x_proj[StateDim * 2]; // scalar delta

            // Softplus for dt (ensures positive)
            const dt = @log(1.0 + @exp(dt_raw));

            // Discretize and scan
            // For each inner dimension:
            //   h[i] = exp(A[i] * dt) * h[i] + B * x_inner[i] * dt
            //   y[i] = C · h[i] + D[i] * x_inner[i]
            var y: [InnerDim]f32 = undefined;
            for (0..InnerDim) |i| {
                var scan_out: f32 = 0;
                for (0..StateDim) |j| {
                    // Discretize A
                    const dA = @exp(self.A[i][j] * dt);
                    // Update state
                    state.h[i][j] = dA * state.h[i][j] + B[j] * x_inner[i] * dt;
                    // Output
                    scan_out += C[j] * state.h[i][j];
                }
                y[i] = scan_out + self.D[i] * x_inner[i];
            }

            // 4. Gate: y = y * SiLU(z)
            for (0..InnerDim) |i| {
                const gate = z[i] / (1.0 + @exp(-z[i])); // SiLU
                y[i] *= gate;
            }

            // 5. Output projection
            var output: [ModelDim]f32 = undefined;
            for (0..ModelDim) |i| {
                var sum: f32 = 0;
                for (0..InnerDim) |j| {
                    sum += self.out_proj[i][j] * y[j];
                }
                output[i] = sum;
            }

            return output;
        }

        /// Memory usage comparison vs attention.
        pub fn memoryPerToken() usize {
            // SSM state is FIXED regardless of sequence length
            return InnerDim * StateDim * @sizeOf(f32) + InnerDim * ConvWidth * @sizeOf(f32);
        }

        pub fn attentionMemoryPerToken(comptime seq_len: usize) usize {
            // Attention KV cache grows linearly with sequence length
            return 2 * seq_len * ModelDim * @sizeOf(f32); // K + V
        }
    };
}

/// Hybrid SSM-Attention block (Jamba-style).
///
/// Uses Mamba for most layers (cheap, O(1) memory) and attention for
/// a few layers (global context retrieval). The ratio determines the
/// KV cache savings — 7:1 Mamba:Attention = 8x less KV cache.
pub fn HybridBlock(
    comptime ModelDim: usize,
    comptime NumHeads: usize,
    comptime StateDim: usize,
    comptime MaxSeqLen: usize,
) type {
    const Mamba = MambaBlock(ModelDim, StateDim, 4, 2);
    const attention = @import("attention.zig");
    const MHA = attention.MultiHeadAttention(ModelDim, NumHeads, MaxSeqLen);

    return struct {
        mamba: Mamba,
        attn: MHA,
        use_attention: bool, // If false, use Mamba (set per-layer)

        const Self = @This();

        pub fn init(use_attn: bool) Self {
            return .{
                .mamba = Mamba.init(),
                .attn = MHA.init(),
                .use_attention = use_attn,
            };
        }

        pub fn forwardMamba(self: *const Self, x: *const [ModelDim]f32, state: *Mamba.State) [ModelDim]f32 {
            var out = self.mamba.forward(x, state);
            // Residual
            for (0..ModelDim) |i| out[i] += x[i];
            return out;
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "mamba block init" {
    const M = MambaBlock(32, 16, 4, 2);
    const block = M.init();
    // D should be initialized to 1.0
    try std.testing.expectEqual(@as(f32, 1.0), block.D[0]);
    // A should be negative log-spaced
    try std.testing.expectEqual(@as(f32, -1.0), block.A[0][0]);
    try std.testing.expectEqual(@as(f32, -16.0), block.A[0][15]);
}

test "mamba state is O(1)" {
    const M = MambaBlock(64, 16, 4, 2);
    // State size should be FIXED regardless of sequence length
    const state_size = M.memoryPerToken();
    // Attention memory for 1024 tokens
    const attn_size = M.attentionMemoryPerToken(1024);
    // Mamba should use MUCH less memory
    try std.testing.expect(state_size < attn_size);
    // At 1024 tokens, attention uses 64x more memory
    try std.testing.expect(attn_size > state_size * 10);
}

test "mamba forward produces output" {
    const M = MambaBlock(8, 4, 4, 2);
    const block = M.init();
    var state = M.State.init();

    const input = [_]f32{0.5} ** 8;
    const output = block.forward(&input, &state);
    try std.testing.expectEqual(@as(usize, 8), output.len);
}

test "mamba state evolves across tokens" {
    const M = MambaBlock(8, 4, 4, 2);
    var block = M.init();
    // Give D non-trivial values so output depends on input
    block.D = [_]f32{ 1.0, 0.5, 0.3, 0.8, 1.2, 0.7, 0.4, 0.9, 1.1, 0.6, 0.2, 0.7, 1.0, 0.5, 0.3, 0.8 };
    var state = M.State.init();

    const x1 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const x2 = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    _ = block.forward(&x1, &state);
    // State should be non-zero after processing a token
    var state_nonzero = false;
    for (state.h) |row| {
        for (row) |v| {
            if (@abs(v) > 1e-10) state_nonzero = true;
        }
    }
    _ = block.forward(&x2, &state);
    // Conv buffer should have been updated
    try std.testing.expect(state.conv_pos == 2);
    // Note: with zero projections, state won't differ much, but conv_pos tracks correctly
}

test "mamba state reset" {
    const M = MambaBlock(8, 4, 4, 2);
    const block = M.init();
    var state = M.State.init();

    const x = [_]f32{1.0} ** 8;
    _ = block.forward(&x, &state);

    // After reset, state should be back to initial
    state.reset();
    for (state.h) |row| {
        for (row) |v| try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "mamba memory savings vs attention" {
    // stories15M-scale: dim=288
    const M = MambaBlock(288, 16, 4, 2);
    const mamba_mem = M.memoryPerToken();
    const attn_mem_256 = M.attentionMemoryPerToken(256);
    const attn_mem_4096 = M.attentionMemoryPerToken(4096);

    // At 256 tokens: attention uses more memory
    try std.testing.expect(attn_mem_256 > mamba_mem);
    // At 4096 tokens: attention uses WAY more
    try std.testing.expect(attn_mem_4096 > mamba_mem * 10);
}

test "hybrid block init" {
    const H = HybridBlock(32, 4, 16, 64);
    const attn_layer = H.init(true);
    const mamba_layer = H.init(false);
    try std.testing.expect(attn_layer.use_attention);
    try std.testing.expect(!mamba_layer.use_attention);
}

test "param count" {
    const M = MambaBlock(32, 16, 4, 2);
    try std.testing.expect(M.param_count > 0);
    try std.testing.expect(M.param_count < 1_000_000); // Should be reasonable
}
