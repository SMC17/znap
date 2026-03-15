// Honest audit benchmark: trained weights, not zero-initialized.
//
// This benchmark addresses the concern that zero-initialized ternary
// weights might be misleadingly fast (all zeros = skip everything).
// We train a policy first, THEN quantize and benchmark.

const std = @import("std");
const znap = @import("znap");

const F = 30;
const H = 64;
const A = 6;

pub fn main() !void {
    std.debug.print(
        \\
        \\  ╔══════════════════════════════════════════════════════════════╗
        \\  ║  znap — HONEST AUDIT BENCHMARK                            ║
        \\  ║  Trained weights, not zero-initialized                     ║
        \\  ║  Platform: {s} / {s}                              ║
        \\  ╚══════════════════════════════════════════════════════════════╝
        \\
        \\
    , .{ @tagName(@import("builtin").cpu.arch), @tagName(@import("builtin").os.tag) });

    // ── Step 1: Train a policy so weights are non-trivial ────────
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var linear = znap.policy.LinearPolicy(F, A).init();
    for (0..10_000) |_| {
        var features: [F]f32 = undefined;
        for (0..F) |i| features[i] = rng.float(f32);
        const action = rng.uintLessThan(usize, A);
        const reward = if (action == 2) @as(f32, 1.0) else @as(f32, -0.3);
        linear.update(&features, action, reward, 0.01);
    }

    var dqn = znap.policy.DQNPolicy(F, H, A).initXavier(42);

    // ── Step 2: Quantize trained weights ─────────────────────────
    const q_linear = znap.quantized.TernaryPolicy(F, A).fromFloat(&linear);
    const q_dqn = znap.quantized.TernaryDQN(F, H, A).fromFloat(&dqn);

    // Check sparsity
    const linear_nz = q_linear.weights.nonZeroCount();
    const linear_total = znap.quantized.TernaryMatrix(A, F).param_count;
    const linear_density = @as(f32, @floatFromInt(linear_nz)) / @as(f32, @floatFromInt(linear_total)) * 100;

    const dqn_w1_nz = q_dqn.w1.nonZeroCount();
    const dqn_w1_total = znap.quantized.TernaryMatrix(H, F).param_count;
    const dqn_w1_density = @as(f32, @floatFromInt(dqn_w1_nz)) / @as(f32, @floatFromInt(dqn_w1_total)) * 100;

    std.debug.print("  Weight density after training + quantization:\n", .{});
    std.debug.print("    Linear: {d}/{d} non-zero ({d:.1}% dense)\n", .{ linear_nz, linear_total, linear_density });
    std.debug.print("    DQN W1: {d}/{d} non-zero ({d:.1}% dense)\n\n", .{ dqn_w1_nz, dqn_w1_total, dqn_w1_density });

    // ── Step 3: Benchmark with trained weights ───────────────────
    const features_const: [F]f32 = blk: {
        var f: [F]f32 = undefined;
        for (0..F) |i| f[i] = @as(f32, @floatFromInt(i)) / @as(f32, F);
        break :blk f;
    };

    std.debug.print("  Benchmarks with TRAINED weights:\n\n", .{});

    // Float linear (trained)
    {
        const warmup = 100_000;
        const iters = 1_000_000;
        for (0..warmup) |_| std.mem.doNotOptimizeAway(linear.evaluate(&features_const));
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |_| std.mem.doNotOptimizeAway(linear.evaluate(&features_const));
        const ns = timer.read();
        const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const ops = @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        std.debug.print("    float linear 30x6 (trained)          {d:>8.3} us  {d:>12.0} ops/s\n", .{ us, ops });
    }

    // Ternary linear (quantized from trained)
    {
        const warmup = 100_000;
        const iters = 1_000_000;
        for (0..warmup) |_| std.mem.doNotOptimizeAway(q_linear.evaluate(&features_const));
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |_| std.mem.doNotOptimizeAway(q_linear.evaluate(&features_const));
        const ns = timer.read();
        const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const ops = @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        std.debug.print("    ternary linear 30x6 (trained)        {d:>8.3} us  {d:>12.0} ops/s\n", .{ us, ops });
    }

    // Float DQN (Xavier-initialized — has non-trivial weights)
    {
        const warmup = 100_000;
        const iters = 1_000_000;
        for (0..warmup) |_| std.mem.doNotOptimizeAway(dqn.evaluate(&features_const));
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |_| std.mem.doNotOptimizeAway(dqn.evaluate(&features_const));
        const ns = timer.read();
        const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const ops = @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        std.debug.print("    float DQN 30x64x6 (Xavier)           {d:>8.3} us  {d:>12.0} ops/s\n", .{ us, ops });
    }

    // Ternary DQN (quantized from Xavier)
    {
        const warmup = 100_000;
        const iters = 1_000_000;
        for (0..warmup) |_| std.mem.doNotOptimizeAway(q_dqn.evaluate(&features_const));
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |_| std.mem.doNotOptimizeAway(q_dqn.evaluate(&features_const));
        const ns = timer.read();
        const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        const ops = @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        std.debug.print("    ternary DQN 30x64x6 (quantized)      {d:>8.3} us  {d:>12.0} ops/s\n", .{ us, ops });
    }

    // ── Step 4: Verify correctness ───────────────────────────────
    std.debug.print("\n  Correctness check (float vs ternary, 1000 random inputs):\n", .{});

    var agree_linear: usize = 0;
    var agree_dqn: usize = 0;

    for (0..1000) |_| {
        var features: [F]f32 = undefined;
        for (0..F) |i| features[i] = rng.float(f32);

        if (linear.evaluate(&features).action == q_linear.evaluate(&features).action) agree_linear += 1;
        if (dqn.evaluate(&features).action == q_dqn.evaluate(&features).action) agree_dqn += 1;
    }

    std.debug.print("    Linear float↔ternary agreement: {d}/1000 ({d:.1}%)\n", .{
        agree_linear,
        @as(f32, @floatFromInt(agree_linear)) / 10.0,
    });
    std.debug.print("    DQN float↔ternary agreement:    {d}/1000 ({d:.1}%)\n", .{
        agree_dqn,
        @as(f32, @floatFromInt(agree_dqn)) / 10.0,
    });

    std.debug.print(
        \\
        \\  ────────────────────────────────────────────────────────────
        \\  This benchmark uses TRAINED, non-trivial weights.
        \\  If ternary is fast with ~67% non-zero weights, it's real.
        \\  ────────────────────────────────────────────────────────────
        \\
    , .{});
}
