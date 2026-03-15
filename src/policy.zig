// Generic policy evaluation for reinforcement learning.
//
// Supports linear (bandit) and 2-layer (DQN) policies with
// comptime-known dimensions. Zero allocations on the hot path.
//
// Configure dimensions at compile time:
//   const MyPolicy = znap.policy.LinearPolicy(64, 8);  // 64 features, 8 actions
//   const result = my_policy.evaluate(&features);

const std = @import("std");
const tensor = @import("tensor.zig");
const simd = @import("simd.zig");

/// Result of a policy evaluation.
pub fn PolicyResult(comptime ActionDim: usize) type {
    return struct {
        action: usize,
        confidence: f32,
        scores: [ActionDim]f32,
        probabilities: [ActionDim]f32,
    };
}

// For backward compat and simple use cases
pub const ActionType = enum(u8) {
    action_0 = 0,
    action_1 = 1,
    action_2 = 2,
    action_3 = 3,
    action_4 = 4,
    action_5 = 5,
    action_6 = 6,
    action_7 = 7,
};

// ── Linear Policy (Bandit) ───────────────────────────────────────

/// Linear policy: scores = W * features + bias, then softmax.
/// Comptime-parameterized over feature and action dimensions.
pub fn LinearPolicy(comptime FeatureDim: usize, comptime ActionDim: usize) type {
    return struct {
        weights: tensor.Tensor(ActionDim, FeatureDim),
        bias: tensor.Vec(ActionDim),

        const Self = @This();
        pub const feature_dim = FeatureDim;
        pub const action_dim = ActionDim;
        pub const param_count = ActionDim * FeatureDim + ActionDim;

        pub fn init() Self {
            return .{
                .weights = tensor.Tensor(ActionDim, FeatureDim).zeros(),
                .bias = tensor.Vec(ActionDim).zeros(),
            };
        }

        pub fn fromWeights(w: [ActionDim * FeatureDim]f32, b: [ActionDim]f32) Self {
            return .{
                .weights = tensor.Tensor(ActionDim, FeatureDim).fromFlat(w),
                .bias = tensor.Vec(ActionDim).fromFlat(b),
            };
        }

        pub fn evaluate(self: *const Self, features: *const [FeatureDim]f32) PolicyResult(ActionDim) {
            var raw = tensor.matvec(ActionDim, FeatureDim, &self.weights, features);
            for (0..ActionDim) |i| raw[i] += self.bias.data[0][i];

            const probs = simd.vsoftmax(ActionDim, &raw);
            const best = simd.vargmax(ActionDim, &raw);

            return .{
                .action = best,
                .confidence = probs[best],
                .scores = raw,
                .probabilities = probs,
            };
        }

        /// Gradient-free bandit update.
        pub fn update(
            self: *Self,
            features: *const [FeatureDim]f32,
            action: usize,
            reward: f32,
            lr: f32,
        ) void {
            const r = self.weights.rowMut(action);
            for (0..FeatureDim) |j| {
                r[j] += lr * reward * features[j];
            }
            self.bias.data[0][action] += lr * reward;

            // Clamp
            for (0..ActionDim) |i| {
                const row = self.weights.rowMut(i);
                for (0..FeatureDim) |j| {
                    row[j] = std.math.clamp(row[j], -3.0, 3.0);
                }
                self.bias.data[0][i] = std.math.clamp(self.bias.data[0][i], -3.0, 3.0);
            }
        }
    };
}

// ── DQN Policy (2-layer) ─────────────────────────────────────────

/// 2-layer neural network: hidden = ReLU(W1*x + b1), output = W2*hidden + b2.
pub fn DQNPolicy(comptime FeatureDim: usize, comptime HiddenDim: usize, comptime ActionDim: usize) type {
    return struct {
        w1: tensor.Tensor(HiddenDim, FeatureDim),
        b1: tensor.Vec(HiddenDim),
        w2: tensor.Tensor(ActionDim, HiddenDim),
        b2: tensor.Vec(ActionDim),

        const Self = @This();
        pub const feature_dim = FeatureDim;
        pub const hidden_dim = HiddenDim;
        pub const action_dim = ActionDim;
        pub const param_count = HiddenDim * FeatureDim + HiddenDim + ActionDim * HiddenDim + ActionDim;

        pub fn init() Self {
            return .{
                .w1 = tensor.Tensor(HiddenDim, FeatureDim).zeros(),
                .b1 = tensor.Vec(HiddenDim).zeros(),
                .w2 = tensor.Tensor(ActionDim, HiddenDim).zeros(),
                .b2 = tensor.Vec(ActionDim).zeros(),
            };
        }

        /// Xavier initialization.
        pub fn initXavier(seed: u64) Self {
            @setEvalBranchQuota(1_000_000);
            var self = Self.init();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            const s1: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(FeatureDim)));
            for (0..HiddenDim) |i| {
                for (0..FeatureDim) |j| {
                    self.w1.data[i][j] = (rng.float(f32) * 2.0 - 1.0) * s1;
                }
            }
            const s2: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(HiddenDim)));
            for (0..ActionDim) |i| {
                for (0..HiddenDim) |j| {
                    self.w2.data[i][j] = (rng.float(f32) * 2.0 - 1.0) * s2;
                }
            }
            return self;
        }

        pub fn evaluate(self: *const Self, features: *const [FeatureDim]f32) PolicyResult(ActionDim) {
            var hidden = tensor.matvec(HiddenDim, FeatureDim, &self.w1, features);
            for (0..HiddenDim) |i| hidden[i] += self.b1.data[0][i];
            hidden = simd.vrelu(HiddenDim, &hidden);

            var output = tensor.matvec(ActionDim, HiddenDim, &self.w2, &hidden);
            for (0..ActionDim) |i| output[i] += self.b2.data[0][i];

            const probs = simd.vsoftmax(ActionDim, &output);
            const best = simd.vargmax(ActionDim, &output);

            return .{
                .action = best,
                .confidence = probs[best],
                .scores = output,
                .probabilities = probs,
            };
        }
    };
}

// ── Ensemble ─────────────────────────────────────────────────────

/// Evaluate N independent policies on the same input.
pub fn Ensemble(comptime N: usize, comptime FeatureDim: usize, comptime ActionDim: usize) type {
    const P = LinearPolicy(FeatureDim, ActionDim);
    return struct {
        agents: [N]P,

        const Self = @This();

        pub fn init() Self {
            return .{ .agents = [_]P{P.init()} ** N };
        }

        pub fn evaluateAll(self: *const Self, features: *const [FeatureDim]f32) [N]PolicyResult(ActionDim) {
            var results: [N]PolicyResult(ActionDim) = undefined;
            for (0..N) |i| results[i] = self.agents[i].evaluate(features);
            return results;
        }

        pub fn totalParams() usize {
            return N * P.param_count;
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "linear policy evaluate" {
    const P = LinearPolicy(8, 4);
    const p = P.init();
    var features: [8]f32 = undefined;
    for (0..8) |i| features[i] = @as(f32, @floatFromInt(i)) / 8.0;

    const r = p.evaluate(&features);
    try std.testing.expect(r.action < 4);
    try std.testing.expect(r.confidence > 0.0);
    try std.testing.expect(r.confidence <= 1.0);

    var prob_sum: f32 = 0;
    for (r.probabilities) |prob| prob_sum += prob;
    try std.testing.expect(@abs(prob_sum - 1.0) < 1e-5);
}

test "bandit update shifts preference" {
    const P = LinearPolicy(4, 3);
    var p = P.init();
    var features = [_]f32{0.5} ** 4;

    for (0..100) |_| p.update(&features, 2, 1.0, 0.05);

    const r = p.evaluate(&features);
    try std.testing.expectEqual(@as(usize, 2), r.action);
    try std.testing.expect(r.confidence > 0.5);
}

test "DQN evaluate" {
    const D = DQNPolicy(8, 32, 4);
    const d = D.initXavier(42);
    var features: [8]f32 = undefined;
    for (0..8) |i| features[i] = @as(f32, @floatFromInt(i)) / 8.0;

    const r = d.evaluate(&features);
    try std.testing.expect(r.action < 4);
    var prob_sum: f32 = 0;
    for (r.probabilities) |prob| prob_sum += prob;
    try std.testing.expect(@abs(prob_sum - 1.0) < 1e-5);
}

test "DQN param count" {
    const D = DQNPolicy(30, 64, 6);
    try std.testing.expectEqual(@as(usize, 2374), D.param_count);
}

test "ensemble evaluates all" {
    const E = Ensemble(5, 8, 4);
    const e = E.init();
    var features = [_]f32{0.5} ** 8;
    const results = e.evaluateAll(&features);
    try std.testing.expectEqual(@as(usize, 5), results.len);
    for (results) |r| try std.testing.expect(r.action < 4);
}

test "generic dimensions" {
    // 128 features, 16 actions — works with any comptime-known shape
    const P = LinearPolicy(128, 16);
    try std.testing.expectEqual(@as(usize, 128 * 16 + 16), P.param_count);

    // 256 features, 32 hidden, 8 actions
    const D = DQNPolicy(256, 32, 8);
    try std.testing.expectEqual(@as(usize, 256 * 32 + 32 + 8 * 32 + 8), D.param_count);
}
