// Deep scaling experiment: Where does DQN ternary quantization become viable?
//
// We know:
//   - Linear policies: 100% ternary agreement at ALL scales (proven)
//   - DQN < 5K params: 40-70% agreement (proven)
//   - DQN > 3B params: ~100% agreement (BitNet paper claim)
//
// This experiment maps the TRANSITION. At what scale does the DQN
// ternary crossover happen? We test from 100 to 500K+ params.
//
// This is the experiment that could produce a NeurIPS figure.
//
// Run: zig build deep-scaling

const std = @import("std");
const bench_timer = @import("bench_timer.zig");
const znap = @import("znap");

fn trainAndQuantize(
    comptime F: usize,
    comptime H: usize,
    comptime A: usize,
    comptime label: []const u8,
    allocator: std.mem.Allocator,
) !void {
    const D = znap.policy.DQNPolicy(F, H, A);
    const params = D.param_count;

    // Generate training data
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const N = @min(params * 10, 50_000); // 10x params or 50K max
    var data = try allocator.alloc([F]f32, N);
    defer allocator.free(data);
    var targets = try allocator.alloc(usize, N);
    defer allocator.free(targets);

    for (0..N) |i| {
        for (0..F) |j| data[i][j] = rng.float(f32);
        var best: usize = 0;
        for (1..@min(A, F)) |a| {
            if (data[i][a] > data[i][best]) best = a;
        }
        targets[i] = best;
    }

    // Train with real backprop
    const result = try znap.gpu_train.trainDQN(F, H, A, data, targets, .{
        .learning_rate = 0.005,
        .epochs = 100,
        .batch_size = @min(64, N / 4),
        .eval_interval = 25,
        .use_margin_loss = false,
        .seed = 42,
    }, allocator);

    // Quantize
    const q_tern = znap.quantized.TernaryDQN(F, H, A).fromFloat(&result.dqn);

    // Evaluate
    var float_correct: usize = 0;
    var tern_correct: usize = 0;
    var agree: usize = 0;
    var margin_sum: f64 = 0;
    const eval_trials = 10_000;

    for (0..eval_trials) |_| {
        var features: [F]f32 = undefined;
        for (0..F) |i| features[i] = rng.float(f32);
        var target: usize = 0;
        for (1..@min(A, F)) |a| {
            if (features[a] > features[target]) target = a;
        }

        const fr = result.dqn.evaluate(&features);
        const tr = q_tern.evaluate(&features);

        if (fr.action == target) float_correct += 1;
        if (tr.action == target) tern_correct += 1;
        if (fr.action == tr.action) agree += 1;

        // Margin
        var second: f32 = -std.math.inf(f32);
        for (0..A) |a| {
            if (a != fr.action and fr.scores[a] > second) second = fr.scores[a];
        }
        margin_sum += fr.scores[fr.action] - second;
    }

    // Benchmark
    const bench_features: [F]f32 = blk: {
        var f: [F]f32 = undefined;
        for (0..F) |i| f[i] = @as(f32, @floatFromInt(i)) / @as(f32, F);
        break :blk f;
    };

    const iters = 100_000;
    for (0..iters / 10) |_| std.mem.doNotOptimizeAway(result.dqn.evaluate(&bench_features));
    var timer = bench_timer.Timer.start() catch unreachable;
    for (0..iters) |_| std.mem.doNotOptimizeAway(result.dqn.evaluate(&bench_features));
    const float_ns = timer.read() / iters;

    for (0..iters / 10) |_| std.mem.doNotOptimizeAway(q_tern.evaluate(&bench_features));
    timer = bench_timer.Timer.start() catch unreachable;
    for (0..iters) |_| std.mem.doNotOptimizeAway(q_tern.evaluate(&bench_features));
    const tern_ns = timer.read() / iters;

    const pct = struct {
        fn f(n: usize, total: usize) f32 {
            return @as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(total)) * 100;
        }
    }.f;
    const avg_margin: f32 = @floatCast(margin_sum / @as(f64, @floatFromInt(eval_trials)));
    const compression = znap.quantized.TernaryMatrix(H, F).compressionRatio();

    std.debug.print("  {s:<16} {d:>7} {d:>7} {d:>7.1}% {d:>7.1}% {d:>7.1}% {d:>7.2} {d:>7} {d:>7} {d:>6.1}x\n", .{
        label,
        params,
        N,
        pct(float_correct, eval_trials),
        pct(tern_correct, eval_trials),
        pct(agree, eval_trials),
        avg_margin,
        float_ns,
        tern_ns,
        compression,
    });
}

pub fn main(init: std.process.Init) !void {
    bench_timer.io = init.io;
    const allocator = init.gpa;

    std.debug.print(
        \\
        \\  +==============================================================================+
        \\  |  znap — Deep Scaling: DQN Ternary Quantization Crossover                    |
        \\  |  Mapping the transition from "broken" to "viable" ternary DQN                |
        \\  |  Platform: {s} / {s}                                                  |
        \\  +==============================================================================+
        \\
        \\  Methodology:
        \\    - Real backprop training (cross-entropy loss, SGD, 100 epochs)
        \\    - 10x params training samples (up to 50K)
        \\    - Post-training ternary quantization
        \\    - 10K eval trials on held-out random inputs
        \\
        \\  Scale            Params TrainN  FloatAcc TernAcc  Agree  Margin  F32ns   Tns  Comp
        \\  ──────────────── ────── ────── ──────── ──────── ────── ────── ────── ────── ──────
        \\
    , .{ @tagName(@import("builtin").cpu.arch), @tagName(@import("builtin").os.tag) });

    // Scale from tiny to large
    try trainAndQuantize(4, 4, 2, "4x4x2", allocator);
    try trainAndQuantize(8, 8, 4, "8x8x4", allocator);
    try trainAndQuantize(8, 16, 4, "8x16x4", allocator);
    try trainAndQuantize(8, 32, 4, "8x32x4", allocator);
    try trainAndQuantize(16, 32, 4, "16x32x4", allocator);
    try trainAndQuantize(16, 64, 4, "16x64x4", allocator);
    try trainAndQuantize(16, 128, 4, "16x128x4", allocator);
    try trainAndQuantize(32, 64, 8, "32x64x8", allocator);
    try trainAndQuantize(32, 128, 8, "32x128x8", allocator);
    try trainAndQuantize(32, 256, 8, "32x256x8", allocator);
    try trainAndQuantize(64, 128, 8, "64x128x8", allocator);
    try trainAndQuantize(64, 256, 8, "64x256x8", allocator);
    try trainAndQuantize(64, 512, 8, "64x512x8", allocator);
    try trainAndQuantize(128, 256, 16, "128x256x16", allocator);
    try trainAndQuantize(128, 512, 16, "128x512x16", allocator);

    std.debug.print(
        \\
        \\  ──────────────── ────── ────── ──────── ──────── ────── ────── ────── ────── ──────
        \\
        \\  This maps the transition curve. Plot Agree% vs log(Params) to find
        \\  the crossover point where ternary DQN becomes viable (>90% agreement).
        \\
        \\  If the crossover is below 100K params, znap can serve ternary DQN
        \\  on CPU. If it's above 1M params, ternary DQN requires GPU-scale models
        \\  and INT8 is the right choice for small-scale deployment.
        \\
        \\  To extend to larger scales (>1M params), use benchmarks/scale_pytorch.py
        \\  on GPU hardware (your Threadripper + 5090 or rented A100/H100).
        \\
    , .{});
}
