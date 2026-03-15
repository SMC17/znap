// DQN Ternary Quantization Crossover Experiment
//
// RESEARCH QUESTION: At what model scale does ternary quantization
// stop degrading DQN quality?
//
// Linear policies show ~100% float<->ternary agreement at all scales.
// DQN showed 42% agreement at 30x64x6. The hidden layer's learned
// representations must reach some threshold of robustness before they
// survive quantization to {-1, 0, +1}.
//
// Two experiments:
//   1. CONTROLLED: Fix task (8 features, 4 actions), vary hidden dim only.
//      Isolates the effect of representation width on quantization survival.
//   2. ORIGINAL SCALES: Test the 8 requested configurations with varying
//      feature/hidden/action dims to match real KAiGR use cases.
//
// Training: Xavier random projection (W1) + bandit output training (W2).
// 30 random W1 seeds, best-of selection, 200K training steps per seed.
//
// Run: zig build dqn-crossover

const std = @import("std");
const znap = @import("znap");

const TRAIN_STEPS: usize = 200_000;
const N_SEEDS: usize = 30;
const EVAL_TRIALS: usize = 10_000;

/// Train a DQN using random projection + bandit output layer.
/// W1 is Xavier-initialized (random feature extractor).
/// W2+b2 are trained using reward-weighted bandit updates.
/// We try N_SEEDS random projections and keep the best.
fn trainDQN(
    comptime F: usize,
    comptime H: usize,
    comptime A: usize,
) znap.policy.DQNPolicy(F, H, A) {
    const DQN = znap.policy.DQNPolicy(F, H, A);

    var best_pol = DQN.init();
    var best_correct: usize = 0;

    for (0..N_SEEDS) |seed_idx| {
        var pol = DQN.initXavier(42 + seed_idx * 7);

        // Zero output layer before training
        for (0..A) |a| {
            for (0..H) |h| pol.w2.data[a][h] = 0;
            pol.b2.data[0][a] = 0;
        }

        // Train W2+b2 as bandit on hidden features
        var prng = std.Random.DefaultPrng.init(100 + seed_idx * 13);
        const rng = prng.random();
        const lr: f32 = 0.003;

        for (0..TRAIN_STEPS) |_| {
            var features: [F]f32 = undefined;
            for (0..F) |i| features[i] = rng.float(f32);

            // Target = argmax of first min(A,F) features
            var target: usize = 0;
            var best_val: f32 = features[0];
            for (1..@min(A, F)) |a| {
                if (features[a] > best_val) {
                    best_val = features[a];
                    target = a;
                }
            }

            // Forward through hidden layer (ReLU activation)
            var hidden = znap.tensor.matvec(H, F, &pol.w1, &features);
            for (0..H) |i| hidden[i] += pol.b1.data[0][i];
            hidden = znap.simd.vrelu(H, &hidden);

            // Compute output scores
            var scores: [A]f32 = undefined;
            for (0..A) |a| {
                var sum: f32 = pol.b2.data[0][a];
                for (0..H) |h| sum += pol.w2.data[a][h] * hidden[h];
                scores[a] = sum;
            }

            var chosen: usize = 0;
            var chosen_val: f32 = scores[0];
            for (1..A) |a| {
                if (scores[a] > chosen_val) {
                    chosen_val = scores[a];
                    chosen = a;
                }
            }

            // Bandit update: reinforce target, penalize wrong choice
            for (0..H) |h| {
                pol.w2.data[target][h] += lr * hidden[h];
                pol.w2.data[target][h] = std.math.clamp(pol.w2.data[target][h], -3.0, 3.0);
            }
            pol.b2.data[0][target] += lr;
            pol.b2.data[0][target] = std.math.clamp(pol.b2.data[0][target], -3.0, 3.0);

            if (chosen != target) {
                for (0..H) |h| {
                    pol.w2.data[chosen][h] -= lr * 0.3 * hidden[h];
                    pol.w2.data[chosen][h] = std.math.clamp(pol.w2.data[chosen][h], -3.0, 3.0);
                }
                pol.b2.data[0][chosen] -= lr * 0.3;
                pol.b2.data[0][chosen] = std.math.clamp(pol.b2.data[0][chosen], -3.0, 3.0);
            }
        }

        // Evaluate this seed with deterministic eval set
        var eval_prng = std.Random.DefaultPrng.init(777);
        const eval_rng = eval_prng.random();
        var correct: usize = 0;
        for (0..2000) |_| {
            var features: [F]f32 = undefined;
            for (0..F) |i| features[i] = eval_rng.float(f32);
            var target: usize = 0;
            var best_val: f32 = features[0];
            for (1..@min(A, F)) |a| {
                if (features[a] > best_val) {
                    best_val = features[a];
                    target = a;
                }
            }
            if (pol.evaluate(&features).action == target) correct += 1;
        }

        if (correct > best_correct) {
            best_correct = correct;
            best_pol = pol;
        }
    }

    return best_pol;
}

/// Evaluate a trained DQN: accuracy, quantization agreement, latency, density.
fn evalAndReport(
    comptime F: usize,
    comptime H: usize,
    comptime A: usize,
    comptime label: []const u8,
    pol: *const znap.policy.DQNPolicy(F, H, A),
) void {
    const DQN = znap.policy.DQNPolicy(F, H, A);
    const TDQN = znap.quantized.TernaryDQN(F, H, A);

    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    var float_correct: usize = 0;
    var float_actions: [EVAL_TRIALS]usize = undefined;
    var eval_inputs: [EVAL_TRIALS][F]f32 = undefined;
    var targets: [EVAL_TRIALS]usize = undefined;

    for (0..EVAL_TRIALS) |t| {
        for (0..F) |i| eval_inputs[t][i] = rng.float(f32);

        var best_action: usize = 0;
        var best_val: f32 = eval_inputs[t][0];
        for (1..@min(A, F)) |a| {
            if (eval_inputs[t][a] > best_val) {
                best_val = eval_inputs[t][a];
                best_action = a;
            }
        }
        targets[t] = best_action;

        const result = pol.evaluate(&eval_inputs[t]);
        float_actions[t] = result.action;
        if (result.action == best_action) float_correct += 1;
    }
    const float_acc = @as(f32, @floatFromInt(float_correct)) / @as(f32, @floatFromInt(EVAL_TRIALS)) * 100.0;

    // Quantize
    const qpol = TDQN.fromFloat(pol);

    var ternary_correct: usize = 0;
    var agree: usize = 0;
    for (0..EVAL_TRIALS) |t| {
        const ternary_action = qpol.evaluate(&eval_inputs[t]).action;
        if (ternary_action == targets[t]) ternary_correct += 1;
        if (float_actions[t] == ternary_action) agree += 1;
    }
    const ternary_acc = @as(f32, @floatFromInt(ternary_correct)) / @as(f32, @floatFromInt(EVAL_TRIALS)) * 100.0;
    const agreement = @as(f32, @floatFromInt(agree)) / @as(f32, @floatFromInt(EVAL_TRIALS)) * 100.0;

    // Latency
    const bench_features: [F]f32 = blk: {
        var f: [F]f32 = undefined;
        for (0..F) |i| f[i] = @as(f32, @floatFromInt(i)) / @as(f32, F);
        break :blk f;
    };

    const iters = 500_000;
    const warmup = 50_000;

    for (0..warmup) |_| std.mem.doNotOptimizeAway(pol.evaluate(&bench_features));
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iters) |_| std.mem.doNotOptimizeAway(pol.evaluate(&bench_features));
    const float_ns = timer.read() / iters;

    for (0..warmup) |_| std.mem.doNotOptimizeAway(qpol.evaluate(&bench_features));
    timer = std.time.Timer.start() catch unreachable;
    for (0..iters) |_| std.mem.doNotOptimizeAway(qpol.evaluate(&bench_features));
    const ternary_ns = timer.read() / iters;

    // Weight density
    const w1_nz = qpol.w1.nonZeroCount();
    const w1_total = znap.quantized.TernaryMatrix(H, F).param_count;
    const w1_density = @as(f32, @floatFromInt(w1_nz)) / @as(f32, @floatFromInt(w1_total)) * 100.0;

    const w2_nz = qpol.w2.nonZeroCount();
    const w2_total = znap.quantized.TernaryMatrix(A, H).param_count;
    const w2_density = @as(f32, @floatFromInt(w2_nz)) / @as(f32, @floatFromInt(w2_total)) * 100.0;

    // ReLU alive %
    var active_neurons: usize = 0;
    const n_test: usize = 1000;
    var prng2 = std.Random.DefaultPrng.init(555);
    const rng2 = prng2.random();
    for (0..n_test) |_| {
        var features: [F]f32 = undefined;
        for (0..F) |i| features[i] = rng2.float(f32);
        var hidden = znap.tensor.matvec(H, F, &pol.w1, &features);
        for (0..H) |i| hidden[i] += pol.b1.data[0][i];
        hidden = znap.simd.vrelu(H, &hidden);
        for (0..H) |i| {
            if (hidden[i] > 0) active_neurons += 1;
        }
    }
    const relu_alive = @as(f32, @floatFromInt(active_neurons)) / @as(f32, @floatFromInt(n_test * H)) * 100.0;

    std.debug.print("  {s:<14} {d:>7} {d:>7.1}% {d:>7.1}% {d:>7.1}% {d:>6} {d:>6} {d:>5.1}% {d:>5.1}% {d:>5.1}%\n", .{
        label,
        DQN.param_count,
        float_acc,
        ternary_acc,
        agreement,
        float_ns,
        ternary_ns,
        w1_density,
        w2_density,
        relu_alive,
    });
}

fn runScale(
    comptime F: usize,
    comptime H: usize,
    comptime A: usize,
    comptime label: []const u8,
) void {
    const pol = trainDQN(F, H, A);
    evalAndReport(F, H, A, label, &pol);
}

pub fn main() !void {
    std.debug.print(
        \\
        \\  +==============================================================================+
        \\  |  znap -- DQN Ternary Quantization Crossover Experiment                       |
        \\  |  Research: At what scale does ternary stop degrading DQN quality?             |
        \\  |  Platform: {s} / {s}                                                   |
        \\  +==============================================================================+
        \\
        \\  EXPERIMENT 1: Controlled (fixed task, variable hidden dim)
        \\  FeatureDim=8, ActionDim=4, HiddenDim varies 4..256
        \\  Training: Xavier W1 (30 seeds) + bandit W2 (200K steps), best-of
        \\  Eval: 10K random inputs, task = argmax of first 4 features
        \\
        \\  Scale          Params  FloatAcc  TernAcc  Agree   F32ns   Tns   W1den  W2den  ReLU%
        \\  -------------- ------- -------- -------- ------- ------ ------ ------ ------ ------
        \\
    , .{ @tagName(@import("builtin").cpu.arch), @tagName(@import("builtin").os.tag) });

    // Experiment 1: Fixed task, variable hidden dim
    runScale(8, 4, 4, "8x4x4");
    runScale(8, 8, 4, "8x8x4");
    runScale(8, 16, 4, "8x16x4");
    runScale(8, 32, 4, "8x32x4");
    runScale(8, 64, 4, "8x64x4");
    runScale(8, 96, 4, "8x96x4");
    runScale(8, 128, 4, "8x128x4");
    runScale(8, 256, 4, "8x256x4");

    std.debug.print(
        \\  -------------- ------- -------- -------- ------- ------ ------ ------ ------ ------
        \\
        \\  EXPERIMENT 2: Original KAiGR configurations
        \\  All dims vary (matching real RL deployment scenarios)
        \\  Same training method, same eval protocol
        \\
        \\  Scale          Params  FloatAcc  TernAcc  Agree   F32ns   Tns   W1den  W2den  ReLU%
        \\  -------------- ------- -------- -------- ------- ------ ------ ------ ------ ------
        \\
    , .{});

    // Experiment 2: Original requested configurations
    runScale(8, 8, 4, "8x8x4");
    runScale(16, 16, 4, "16x16x4");
    runScale(16, 32, 4, "16x32x4");
    runScale(30, 32, 6, "30x32x6");
    runScale(30, 64, 6, "30x64x6");
    runScale(64, 64, 8, "64x64x8");
    runScale(64, 128, 8, "64x128x8");
    runScale(128, 128, 16, "128x128x16");

    std.debug.print(
        \\  -------------- ------- -------- -------- ------- ------ ------ ------ ------ ------
        \\
        \\  Legend:
        \\    Params   = total trainable parameters (W1 + b1 + W2 + b2)
        \\    FloatAcc = float32 DQN accuracy on argmax task (%)
        \\    TernAcc  = ternary DQN accuracy on same task (%)
        \\    Agree    = float<->ternary action agreement (%)
        \\    F32ns    = float32 inference latency (nanoseconds)
        \\    Tns      = ternary inference latency (nanoseconds)
        \\    W1den    = layer 1 ternary weight density (non-zero %)
        \\    W2den    = layer 2 ternary weight density (non-zero %)
        \\    ReLU%    = fraction of hidden neurons active (not killed by ReLU)
        \\
        \\  KEY FINDINGS:
        \\    Experiment 1 isolates the hidden-dim effect on quantization survival.
        \\    Experiment 2 shows real-world DQN configurations from KAiGR.
        \\
        \\    The crossover point (if any) where Agree > 90% reveals the minimum
        \\    DQN size where ternary quantization becomes production-viable.
        \\
        \\    If agreement stays low even at large hidden dims, it means DQN
        \\    quantization requires quantization-aware training -- not just
        \\    post-training quantization of learned weights.
        \\
    , .{});
}
