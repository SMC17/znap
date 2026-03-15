// Training loop for znap policies.
//
// Implements gradient-free and simple gradient-based training for small models.
// Not meant to compete with PyTorch for large-scale training — this is for
// fine-tuning small policies (< 10K params) directly on the inference hardware.
//
// Training strategies:
//   - Bandit: gradient-free weight updates from reward signals
//   - Evolution: population-based training with tournament selection
//   - Cross-entropy method (CEM): elite fraction optimization
//   - Simple SGD: numerical gradient estimation for DQN

const std = @import("std");
const policy = @import("policy.zig");
const replay = @import("replay.zig");
const simd = @import("simd.zig");

// ── Training Configuration ───────────────────────────────────────

pub const TrainConfig = struct {
    learning_rate: f32 = 0.01,
    batch_size: usize = 32,
    max_steps: usize = 10_000,
    eval_interval: usize = 100,
    /// For evolution/CEM strategies.
    population_size: usize = 32,
    elite_fraction: f32 = 0.25,
    /// Noise scale for gradient estimation.
    noise_scale: f32 = 0.01,
    /// Random seed.
    seed: u64 = 42,
};

pub const TrainResult = struct {
    steps_completed: usize,
    final_reward: f32,
    best_reward: f32,
    avg_reward_history: [1000]f32,
    history_len: usize,
};

// ── Bandit Training ──────────────────────────────────────────────

/// Train a linear policy from a replay buffer using bandit updates.
/// This is the simplest and fastest training method.
pub fn trainBandit(
    comptime F: usize,
    comptime A: usize,
    pol: *policy.LinearPolicy(F, A),
    buf: anytype,
    config: TrainConfig,
) TrainResult {
    var prng = std.Random.DefaultPrng.init(config.seed);
    var rng = prng.random();
    var result = TrainResult{
        .steps_completed = 0,
        .final_reward = 0,
        .best_reward = -std.math.inf(f32),
        .avg_reward_history = [_]f32{0} ** 1000,
        .history_len = 0,
    };

    var reward_sum: f32 = 0;
    var reward_count: usize = 0;

    const steps = @min(config.max_steps, buf.count * 10);
    for (0..steps) |step| {
        const t = buf.samplePrioritized(&rng);
        pol.update(&t.state, t.action, t.reward, config.learning_rate);

        reward_sum += t.reward;
        reward_count += 1;

        if (step > 0 and step % config.eval_interval == 0) {
            const avg = reward_sum / @as(f32, @floatFromInt(reward_count));
            if (result.history_len < 1000) {
                result.avg_reward_history[result.history_len] = avg;
                result.history_len += 1;
            }
            if (avg > result.best_reward) result.best_reward = avg;
            reward_sum = 0;
            reward_count = 0;
        }

        result.steps_completed = step + 1;
    }

    result.final_reward = if (reward_count > 0)
        reward_sum / @as(f32, @floatFromInt(reward_count))
    else
        result.best_reward;

    return result;
}

// ── Evolution Strategy ───────────────────────────────────────────

/// Evolutionary training: maintain a population of policies,
/// evaluate them, keep the best, mutate to create the next generation.
pub fn EvolutionTrainer(comptime F: usize, comptime A: usize) type {
    const P = policy.LinearPolicy(F, A);
    const MAX_POP = 64;

    return struct {
        population: [MAX_POP]P,
        fitness: [MAX_POP]f32,
        pop_size: usize,
        generation: usize,
        best_fitness: f32,

        const Self = @This();

        pub fn init(pop_size: usize, seed: u64) Self {
            var self: Self = undefined;
            self.pop_size = @min(pop_size, MAX_POP);
            self.generation = 0;
            self.best_fitness = -std.math.inf(f32);

            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            for (0..self.pop_size) |i| {
                self.population[i] = P.init();
                self.fitness[i] = 0;
                // Random initialization with small weights
                for (0..A) |a| {
                    const row = self.population[i].weights.rowMut(a);
                    for (0..F) |f| {
                        row[f] = (rng.float(f32) * 2.0 - 1.0) * 0.1;
                    }
                }
            }

            return self;
        }

        /// Evaluate fitness for each member using a fitness function.
        /// The fitness function takes a policy pointer and returns a reward.
        pub fn evaluate(self: *Self, fitness_fn: *const fn (*const P) f32) void {
            for (0..self.pop_size) |i| {
                self.fitness[i] = fitness_fn(&self.population[i]);
            }
        }

        /// Select elites, mutate, create next generation.
        pub fn evolve(self: *Self, elite_fraction: f32, noise_scale: f32, seed: u64) void {
            const elite_count = @max(@as(usize, @intFromFloat(@as(f32, @floatFromInt(self.pop_size)) * elite_fraction)), 1);

            // Sort by fitness (descending) — simple selection sort
            for (0..self.pop_size) |i| {
                var best_idx = i;
                for (i + 1..self.pop_size) |j| {
                    if (self.fitness[j] > self.fitness[best_idx]) best_idx = j;
                }
                if (best_idx != i) {
                    const tmp_p = self.population[i];
                    self.population[i] = self.population[best_idx];
                    self.population[best_idx] = tmp_p;
                    const tmp_f = self.fitness[i];
                    self.fitness[i] = self.fitness[best_idx];
                    self.fitness[best_idx] = tmp_f;
                }
            }

            self.best_fitness = self.fitness[0];
            self.generation += 1;

            // Mutate: fill non-elite slots with mutated copies of elites
            var prng = std.Random.DefaultPrng.init(seed +% self.generation);
            const rng = prng.random();

            for (elite_count..self.pop_size) |i| {
                const parent_idx = i % elite_count;
                self.population[i] = self.population[parent_idx];

                // Add Gaussian-ish noise
                for (0..A) |a| {
                    const row = self.population[i].weights.rowMut(a);
                    for (0..F) |f| {
                        row[f] += (rng.float(f32) * 2.0 - 1.0) * noise_scale;
                        row[f] = std.math.clamp(row[f], -3.0, 3.0);
                    }
                    self.population[i].bias.data[0][a] += (rng.float(f32) * 2.0 - 1.0) * noise_scale * 0.1;
                }
            }
        }

        /// Get the best policy from the current population.
        pub fn best(self: *const Self) *const P {
            return &self.population[0]; // Sorted after evolve()
        }
    };
}

// ── Numerical Gradient SGD for DQN ───────────────────────────────

/// Estimate the gradient of a loss function with respect to policy parameters
/// using finite differences. Simple but works for small models.
pub fn numericalGradientStep(
    comptime F: usize,
    comptime A: usize,
    pol: *policy.LinearPolicy(F, A),
    loss_fn: *const fn (*const policy.LinearPolicy(F, A)) f32,
    lr: f32,
    epsilon: f32,
) f32 {
    const base_loss = loss_fn(pol);

    // Perturb each weight and estimate gradient
    for (0..A) |a| {
        const row = pol.weights.rowMut(a);
        for (0..F) |f| {
            const orig = row[f];

            // Forward difference
            row[f] = orig + epsilon;
            const loss_plus = loss_fn(pol);

            // Estimate gradient
            const grad = (loss_plus - base_loss) / epsilon;

            // SGD update
            row[f] = orig - lr * grad;
            row[f] = std.math.clamp(row[f], -3.0, 3.0);
        }

        // Bias gradient
        const orig_b = pol.bias.data[0][a];
        pol.bias.data[0][a] = orig_b + epsilon;
        const loss_plus_b = loss_fn(pol);
        const grad_b = (loss_plus_b - base_loss) / epsilon;
        pol.bias.data[0][a] = orig_b - lr * grad_b;
        pol.bias.data[0][a] = std.math.clamp(pol.bias.data[0][a], -3.0, 3.0);
    }

    return base_loss;
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "bandit training improves reward" {
    const P = policy.LinearPolicy(4, 3);
    var pol = P.init();

    // Create a buffer with biased rewards
    var buf = replay.ReplayBuffer(1000, 4).init();
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    for (0..500) |_| {
        var state = [_]f32{0.5} ** 4;
        for (0..4) |i| state[i] = rng.float(f32);
        // Action 1 always gets high reward
        buf.record(state, 1, 1.0, state, false, 0);
        // Other actions get low reward
        buf.record(state, 0, -0.5, state, false, 0);
        buf.record(state, 2, -0.3, state, false, 0);
    }

    const result = trainBandit(4, 3, &pol, &buf, .{
        .learning_rate = 0.05,
        .max_steps = 2000,
        .eval_interval = 100,
    });

    try std.testing.expect(result.steps_completed > 0);

    // After training, policy should prefer action 1
    var test_features = [_]f32{0.5} ** 4;
    const eval = pol.evaluate(&test_features);
    try std.testing.expectEqual(@as(usize, 1), eval.action);
}

test "evolution trainer init" {
    const E = EvolutionTrainer(4, 3);
    const trainer = E.init(16, 42);
    try std.testing.expectEqual(@as(usize, 16), trainer.pop_size);
    try std.testing.expectEqual(@as(usize, 0), trainer.generation);
}

test "evolution improves over generations" {
    const P = policy.LinearPolicy(4, 3);
    const E = EvolutionTrainer(4, 3);
    var trainer = E.init(16, 42);

    // Fitness = how much the policy prefers action 1
    const fitnessFn = struct {
        fn f(p: *const P) f32 {
            var features = [_]f32{0.5} ** 4;
            const result = p.evaluate(&features);
            return if (result.action == 1) result.confidence else -result.confidence;
        }
    }.f;

    var best_gen0: f32 = -999;
    trainer.evaluate(fitnessFn);
    best_gen0 = trainer.best_fitness;

    // Run 20 generations
    for (0..20) |_| {
        trainer.evaluate(fitnessFn);
        trainer.evolve(0.25, 0.1, 42);
    }

    // Best fitness should improve
    try std.testing.expect(trainer.best_fitness >= best_gen0);
}

test "numerical gradient reduces loss" {
    const P = policy.LinearPolicy(4, 3);
    var pol = P.init();

    // Loss = negative of confidence for action 1
    const lossFn = struct {
        fn f(p: *const P) f32 {
            var features = [_]f32{0.5} ** 4;
            const result = p.evaluate(&features);
            return -result.probabilities[1]; // We want action 1
        }
    }.f;

    const initial_loss = lossFn(&pol);

    // Take gradient steps
    for (0..50) |_| {
        _ = numericalGradientStep(4, 3, &pol, lossFn, 0.1, 0.001);
    }

    const final_loss = lossFn(&pol);
    try std.testing.expect(final_loss < initial_loss);
}

test "train config defaults" {
    const cfg = TrainConfig{};
    try std.testing.expectEqual(@as(f32, 0.01), cfg.learning_rate);
    try std.testing.expectEqual(@as(usize, 32), cfg.batch_size);
    try std.testing.expectEqual(@as(usize, 10_000), cfg.max_steps);
}
