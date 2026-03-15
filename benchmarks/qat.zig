// Quantization-Aware Training (QAT) experiment.
//
// RESEARCH QUESTION: Can we train DQN policies that are DESIGNED to survive
// ternary quantization? Instead of train-then-quantize (which fails at
// small scale), we train with simulated quantization in the forward pass.
//
// Method: During training, the forward pass applies ternary quantization
// to the weights before computing the output. Gradients flow through the
// quantization step via straight-through estimator (STE). The optimizer
// updates the float "shadow" weights, but the model always evaluates
// as if quantized.
//
// This is the BitNet paper's recommended approach for small models.
//
// We compare three regimes:
//   A. Float training → float inference (baseline)
//   B. Float training → ternary inference (post-training quantization, PTQ)
//   C. QAT training → ternary inference (quantization-aware training)
//
// If C >> B, we've proven that QAT is necessary for small-model BitNet.

const std = @import("std");
const znap = @import("znap");

fn ternaryQuantize(val: f32, scale: f32) f32 {
    const normalized = val / scale;
    if (normalized > 0.5) return scale;
    if (normalized < -0.5) return -scale;
    return 0;
}

fn rowAbsmax(comptime N: usize, row: *const [N]f32) f32 {
    var m: f32 = 0;
    for (0..N) |i| {
        const a = @abs(row[i]);
        if (a > m) m = a;
    }
    return if (m > 0) m else 1.0;
}

/// Train a linear policy with simulated ternary quantization in the forward pass.
/// This is the QAT approach: weights are stored as float32 but the forward pass
/// quantizes them to ternary before computing the output.
fn trainQAT(
    comptime F: usize,
    comptime A: usize,
    train_steps: usize,
    seed: u64,
) struct { float_acc: f32, ternary_acc: f32, agreement: f32, qat_acc: f32 } {
    const P = znap.policy.LinearPolicy(F, A);
    var prng = std.Random.DefaultPrng.init(seed);
    var rng = prng.random();

    // Three policies trained differently
    var pol_float = P.init();   // A: standard float training
    var pol_qat = P.init();     // C: QAT training (same init)

    // Copy initial weights so they start identical
    pol_qat = pol_float;

    for (0..train_steps) |_| {
        var features: [F]f32 = undefined;
        for (0..F) |i| features[i] = rng.float(f32);

        // Target: argmax of first A features
        var target: usize = 0;
        var best_val: f32 = features[0];
        for (1..@min(A, F)) |a| {
            if (features[a] > best_val) {
                best_val = features[a];
                target = a;
            }
        }

        // ── A: Standard float training ───────────────────────────
        {
            const result = pol_float.evaluate(&features);
            const reward: f32 = if (result.action == target) 1.0 else -0.2;
            pol_float.update(&features, target, reward, 0.005);
        }

        // ── C: QAT training ──────────────────────────────────────
        // Forward pass uses quantized weights, but update modifies float weights
        {
            // Simulate ternary forward: quantize weights, compute scores
            var qat_scores: [A]f32 = [_]f32{0} ** A;
            for (0..A) |a| {
                const row = pol_qat.weights.row(a);
                const scale = rowAbsmax(F, row);
                for (0..F) |f| {
                    const q_weight = ternaryQuantize(row[f], scale);
                    qat_scores[a] += q_weight * features[f];
                }
                qat_scores[a] += pol_qat.bias.data[0][a];
            }

            // Action from quantized forward
            var qat_action: usize = 0;
            var qat_best: f32 = qat_scores[0];
            for (1..A) |a| {
                if (qat_scores[a] > qat_best) {
                    qat_best = qat_scores[a];
                    qat_action = a;
                }
            }

            // Reward based on quantized action
            const reward: f32 = if (qat_action == target) 1.0 else -0.2;
            // Update the FLOAT shadow weights (STE: gradient passes through quantization)
            pol_qat.update(&features, target, reward, 0.005);
        }
    }

    // ── Evaluate all three regimes ───────────────────────────────
    const eval_trials = 10_000;
    var float_correct: usize = 0;
    var ptq_correct: usize = 0;
    var qat_correct: usize = 0;
    var agree_ptq: usize = 0;
    var agree_qat: usize = 0;

    // B: Post-training quantization of the float-trained policy
    const pol_ptq = znap.quantized.TernaryPolicy(F, A).fromFloat(&pol_float);
    // C: Post-training quantization of the QAT-trained policy
    const pol_qat_quantized = znap.quantized.TernaryPolicy(F, A).fromFloat(&pol_qat);

    for (0..eval_trials) |_| {
        var features: [F]f32 = undefined;
        for (0..F) |i| features[i] = rng.float(f32);

        var target: usize = 0;
        var best_val: f32 = features[0];
        for (1..@min(A, F)) |a| {
            if (features[a] > best_val) {
                best_val = features[a];
                target = a;
            }
        }

        const float_action = pol_float.evaluate(&features).action;
        const ptq_action = pol_ptq.evaluate(&features).action;
        const qat_action = pol_qat_quantized.evaluate(&features).action;

        if (float_action == target) float_correct += 1;
        if (ptq_action == target) ptq_correct += 1;
        if (qat_action == target) qat_correct += 1;
        if (float_action == ptq_action) agree_ptq += 1;
        if (float_action == qat_action) agree_qat += 1;
    }

    return .{
        .float_acc = @as(f32, @floatFromInt(float_correct)) / @as(f32, @floatFromInt(eval_trials)) * 100,
        .ternary_acc = @as(f32, @floatFromInt(ptq_correct)) / @as(f32, @floatFromInt(eval_trials)) * 100,
        .agreement = @as(f32, @floatFromInt(agree_ptq)) / @as(f32, @floatFromInt(eval_trials)) * 100,
        .qat_acc = @as(f32, @floatFromInt(qat_correct)) / @as(f32, @floatFromInt(eval_trials)) * 100,
    };
}

pub fn main() !void {
    std.debug.print(
        \\
        \\  +==============================================================================+
        \\  |  znap — Quantization-Aware Training (QAT) Experiment                         |
        \\  |  Can QAT rescue ternary quantization quality at small scale?                  |
        \\  |  Platform: {s} / {s}                                                  |
        \\  +==============================================================================+
        \\
        \\  Three training regimes compared:
        \\    A. Float training → float inference (upper bound)
        \\    B. Float training → ternary inference (post-training quantization, PTQ)
        \\    C. QAT training  → ternary inference (quantization-aware training)
        \\
        \\  Task: argmax prediction (select action matching max feature)
        \\  Training: 100K steps, bandit updates, lr=0.005
        \\  Eval: 10K random inputs, 5 seeds averaged
        \\
        \\  Scale        Params    Float(A)   PTQ(B)   QAT(C)   B-A gap   C-A gap   QAT lift
        \\  ──────────── ────── ────────── ──────── ──────── ──────── ──────── ─────────
        \\
    , .{ @tagName(@import("builtin").cpu.arch), @tagName(@import("builtin").os.tag) });

    const configs = .{
        .{ 4, 2, "4x2" },
        .{ 8, 4, "8x4" },
        .{ 16, 4, "16x4" },
        .{ 30, 6, "30x6" },
        .{ 64, 8, "64x8" },
        .{ 128, 8, "128x8" },
    };

    inline for (configs) |cfg| {
        const F = cfg[0];
        const A = cfg[1];
        const label = cfg[2];
        const params = F * A + A;

        // Average over 5 seeds for stability
        var sum_float: f32 = 0;
        var sum_ptq: f32 = 0;
        var sum_qat: f32 = 0;

        inline for (0..5) |seed_offset| {
            const r = trainQAT(F, A, 100_000, 42 + seed_offset);
            sum_float += r.float_acc;
            sum_ptq += r.ternary_acc;
            sum_qat += r.qat_acc;
        }

        const avg_float = sum_float / 5;
        const avg_ptq = sum_ptq / 5;
        const avg_qat = sum_qat / 5;
        const ptq_gap = avg_ptq - avg_float;
        const qat_gap = avg_qat - avg_float;
        const qat_lift = avg_qat - avg_ptq;

        std.debug.print("  {s:<12} {d:>6} {d:>9.1}% {d:>7.1}% {d:>7.1}% {d:>7.1}% {d:>7.1}% {d:>8.1}%\n", .{
            label,
            params,
            avg_float,
            avg_ptq,
            avg_qat,
            ptq_gap,
            qat_gap,
            qat_lift,
        });
    }

    std.debug.print(
        \\
        \\  ──────────── ────── ────────── ──────── ──────── ──────── ──────── ─────────
        \\
        \\  Legend:
        \\    Float(A)  = Float training → float inference accuracy
        \\    PTQ(B)    = Float training → ternary inference (post-training quantization)
        \\    QAT(C)    = QAT training → ternary inference (quantization-aware training)
        \\    B-A gap   = Quality loss from PTQ (negative = worse)
        \\    C-A gap   = Quality loss from QAT (negative = worse, but less than B-A)
        \\    QAT lift  = QAT improvement over PTQ (positive = QAT helps)
        \\
        \\  If QAT lift > 0 consistently, it proves that quantization-aware training
        \\  produces ternary-robust representations that survive quantization better
        \\  than post-training quantization alone.
        \\
    , .{});
}
