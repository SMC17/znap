// Mixture of Experts (DeepSeek-style).
//
// Implements sparse MoE with:
//   - Router: linear → softmax → top-k expert selection
//   - Expert pool: N experts, each a small SwiGLU FFN
//   - Shared expert: always active, captures cross-context knowledge
//   - Sparse dispatch: only compute active experts (massive FLOP savings)
//
// The key insight: scaling parameter count without scaling compute.
// With 64 experts and top-2 routing, each token only uses ~3% of params
// but the model has the capacity of a much larger dense model.
//
// Reference: DeepSeek-V3 Technical Report, Section 3.2

const std = @import("std");
const simd = @import("simd.zig");
const tensor = @import("tensor.zig");

/// A single expert: small SwiGLU feed-forward network.
pub fn Expert(comptime ModelDim: usize, comptime HiddenDim: usize) type {
    return struct {
        w1: tensor.Tensor(HiddenDim, ModelDim), // Up projection
        w_gate: tensor.Tensor(HiddenDim, ModelDim), // Gate projection
        w2: tensor.Tensor(ModelDim, HiddenDim), // Down projection

        const Self = @This();
        pub const model_dim = ModelDim;
        pub const hidden_dim = HiddenDim;
        pub const param_count = 3 * HiddenDim * ModelDim;

        pub fn init() Self {
            return .{
                .w1 = tensor.Tensor(HiddenDim, ModelDim).zeros(),
                .w_gate = tensor.Tensor(HiddenDim, ModelDim).zeros(),
                .w2 = tensor.Tensor(ModelDim, HiddenDim).zeros(),
            };
        }

        /// SwiGLU forward: (W1·x ⊙ SiLU(W_gate·x)) · W2
        pub fn forward(self: *const Self, x: *const [ModelDim]f32) [ModelDim]f32 {
            var up = tensor.matvec(HiddenDim, ModelDim, &self.w1, x);
            var gate = tensor.matvec(HiddenDim, ModelDim, &self.w_gate, x);

            // SiLU(gate) = gate * sigmoid(gate), then element-wise multiply
            for (0..HiddenDim) |i| {
                gate[i] = gate[i] / (1.0 + @exp(-gate[i]));
                up[i] *= gate[i];
            }

            return tensor.matvec(ModelDim, HiddenDim, &self.w2, &up);
        }
    };
}

/// Mixture of Experts layer with top-k routing and shared experts.
///
/// ModelDim:        embedding dimension
/// NumExperts:      total number of routed experts
/// TopK:            number of experts activated per token
/// ExpertHiddenDim: hidden dimension of each expert FFN
/// NumShared:       number of shared (always-active) experts
pub fn MoELayer(
    comptime ModelDim: usize,
    comptime NumExperts: usize,
    comptime TopK: usize,
    comptime ExpertHiddenDim: usize,
    comptime NumShared: usize,
) type {
    const ExpertType = Expert(ModelDim, ExpertHiddenDim);

    return struct {
        // Router: projects input to expert scores
        router: tensor.Tensor(NumExperts, ModelDim),

        // Expert pool
        experts: [NumExperts]ExpertType,

        // Shared experts (always active)
        shared_experts: [NumShared]ExpertType,

        const Self = @This();
        pub const model_dim = ModelDim;
        pub const num_experts = NumExperts;
        pub const top_k = TopK;
        pub const expert_hidden_dim = ExpertHiddenDim;
        pub const num_shared = NumShared;

        pub const router_param_count = NumExperts * ModelDim;
        pub const expert_param_count = NumExperts * ExpertType.param_count;
        pub const shared_param_count = NumShared * ExpertType.param_count;
        pub const param_count = router_param_count + expert_param_count + shared_param_count;

        /// Active compute per token (only TopK routed + shared).
        pub const active_param_count = (TopK + NumShared) * ExpertType.param_count + router_param_count;

        pub fn init() Self {
            return .{
                .router = tensor.Tensor(NumExperts, ModelDim).zeros(),
                .experts = [_]ExpertType{ExpertType.init()} ** NumExperts,
                .shared_experts = [_]ExpertType{ExpertType.init()} ** NumShared,
            };
        }

        /// Route result: which experts are selected and their weights.
        pub const RouteResult = struct {
            indices: [TopK]usize,
            weights: [TopK]f32,
        };

        /// Compute routing: linear → softmax → top-k selection.
        pub fn route(self: *const Self, x: *const [ModelDim]f32) RouteResult {
            // Router logits
            var logits = tensor.matvec(NumExperts, ModelDim, &self.router, x);

            // Softmax over expert scores
            var max_val: f32 = logits[0];
            for (1..NumExperts) |i| {
                if (logits[i] > max_val) max_val = logits[i];
            }
            var sum_exp: f32 = 0;
            for (0..NumExperts) |i| {
                logits[i] = @exp(logits[i] - max_val);
                sum_exp += logits[i];
            }
            for (0..NumExperts) |i| logits[i] /= sum_exp;

            // Top-K selection
            var result: RouteResult = undefined;
            var selected: [NumExperts]bool = [_]bool{false} ** NumExperts;

            for (0..TopK) |k| {
                var best_idx: usize = 0;
                var best_val: f32 = -std.math.inf(f32);
                for (0..NumExperts) |i| {
                    if (!selected[i] and logits[i] > best_val) {
                        best_val = logits[i];
                        best_idx = i;
                    }
                }
                result.indices[k] = best_idx;
                result.weights[k] = best_val;
                selected[best_idx] = true;
            }

            // Renormalize weights of selected experts
            var weight_sum: f32 = 0;
            for (0..TopK) |k| weight_sum += result.weights[k];
            if (weight_sum > 0) {
                for (0..TopK) |k| result.weights[k] /= weight_sum;
            }

            return result;
        }

        /// Forward pass: route → sparse dispatch → weighted combine + shared.
        pub fn forward(self: *const Self, x: *const [ModelDim]f32) [ModelDim]f32 {
            // Route to top-k experts
            const routing = self.route(x);

            // Initialize output
            var output: [ModelDim]f32 = [_]f32{0} ** ModelDim;

            // Sparse dispatch: only compute selected experts
            for (0..TopK) |k| {
                const expert_idx = routing.indices[k];
                const weight = routing.weights[k];
                const expert_out = self.experts[expert_idx].forward(x);

                for (0..ModelDim) |d| {
                    output[d] += weight * expert_out[d];
                }
            }

            // Add shared expert contributions (always active, unweighted)
            for (0..NumShared) |s| {
                const shared_out = self.shared_experts[s].forward(x);
                for (0..ModelDim) |d| {
                    output[d] += shared_out[d];
                }
            }

            return output;
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "Expert SwiGLU forward zero weights" {
    const E = Expert(8, 16);
    const expert = E.init();
    const x = [_]f32{0.5} ** 8;
    const out = expert.forward(&x);
    // Zero weights → zero output
    for (out) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
}

test "Expert SwiGLU forward non-zero" {
    const E = Expert(4, 8);
    var expert = E.init();

    // Set gate weights to produce positive gates, w1 to pass through
    for (0..4) |i| {
        expert.w_gate.data[i][i] = 2.0; // Strong positive → sigmoid ≈ 1
        expert.w1.data[i][i] = 1.0;
        expert.w2.data[i][i] = 1.0;
    }

    const x = [_]f32{ 1.0, 0.5, 0.3, 0.1 };
    const out = expert.forward(&x);

    // Output should be non-zero
    var any_nonzero = false;
    for (out) |v| {
        if (@abs(v) > 1e-6) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
}

test "MoE routing selects top-k" {
    const MoE = MoELayer(8, 4, 2, 16, 1);
    var moe = MoE.init();

    // Set router weights so expert 1 and 3 are preferred for specific inputs
    moe.router.data[1][0] = 10.0;
    moe.router.data[3][1] = 10.0;

    const x = [_]f32{ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const routing = moe.route(&x);

    // Top-2 should select experts 1 and 3 (highest logits)
    var has_1 = false;
    var has_3 = false;
    for (0..2) |k| {
        if (routing.indices[k] == 1) has_1 = true;
        if (routing.indices[k] == 3) has_3 = true;
    }
    try std.testing.expect(has_1);
    try std.testing.expect(has_3);

    // Weights should sum to 1 (renormalized)
    var wsum: f32 = 0;
    for (0..2) |k| wsum += routing.weights[k];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wsum, 1e-5);
}

test "MoE forward produces output" {
    const MoE = MoELayer(8, 4, 2, 16, 1);
    const moe = MoE.init();

    const x = [_]f32{0.1} ** 8;
    const out = moe.forward(&x);

    // With zero weights, output should be zero
    try std.testing.expectEqual(@as(usize, 8), out.len);
    for (out) |v| {
        try std.testing.expect(!std.math.isNan(v));
    }
}

test "MoE param count and sparsity" {
    const MoE = MoELayer(64, 8, 2, 128, 1);

    // Total params: router + 8 experts + 1 shared
    const expert_params = 3 * 128 * 64; // per expert
    try std.testing.expectEqual(@as(usize, 8 * 64), MoE.router_param_count);
    try std.testing.expectEqual(@as(usize, 8 * expert_params), MoE.expert_param_count);
    try std.testing.expectEqual(@as(usize, 1 * expert_params), MoE.shared_param_count);

    // Active params should be much less than total
    // Active: router + 2 routed + 1 shared = router + 3*expert_params
    try std.testing.expect(MoE.active_param_count < MoE.param_count);

    // Sparsity: active/total ratio
    const sparsity = @as(f32, @floatFromInt(MoE.active_param_count)) / @as(f32, @floatFromInt(MoE.param_count));
    try std.testing.expect(sparsity < 0.5); // Less than half of params active
}

test "MoE shared expert always contributes" {
    const MoE = MoELayer(4, 4, 1, 8, 1);
    var moe = MoE.init();

    // Set ONLY the shared expert to have non-zero weights
    for (0..4) |i| {
        moe.shared_experts[0].w_gate.data[i][i] = 2.0;
        moe.shared_experts[0].w1.data[i][i] = 1.0;
        moe.shared_experts[0].w2.data[i][i] = 1.0;
    }

    const x = [_]f32{ 1.0, 0.5, 0.3, 0.1 };
    const out = moe.forward(&x);

    // Shared expert should contribute even though routed experts are zero
    var any_nonzero = false;
    for (out) |v| {
        if (@abs(v) > 1e-6) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
}
