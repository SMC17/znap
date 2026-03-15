// Example: Training and evaluating an RL agent with znap.
//
// This demonstrates the full training loop:
//   1. Initialize a policy (linear or DQN)
//   2. Collect experience into a replay buffer
//   3. Update the policy from experience
//   4. Quantize to BitNet 1.58-bit for deployment
//   5. Compare float vs quantized inference

const std = @import("std");
const znap = @import("znap");

const FEATURES = 16;
const ACTIONS = 4;

pub fn main() !void {
    std.debug.print(
        \\
        \\  znap — RL Agent Training Example
        \\  ─────────────────────────────────
        \\  Features: {d}  Actions: {d}
        \\
        \\
    , .{ FEATURES, ACTIONS });

    // ── Step 1: Initialize ────────────────────────────────────────
    var policy = znap.policy.LinearPolicy(FEATURES, ACTIONS).init();
    var buffer = znap.replay.ReplayBuffer(10_000, FEATURES).init();
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    std.debug.print("  [1/5] Policy initialized: {d} params, {d} bytes\n", .{
        @TypeOf(policy).param_count,
        @TypeOf(policy).param_count * 4,
    });

    // ── Step 2: Collect experience ───────────────────────────────
    const episodes = 1000;
    const steps_per_episode = 20;

    for (0..episodes) |ep| {
        var state: [FEATURES]f32 = undefined;
        for (0..FEATURES) |i| state[i] = rng.float(f32);

        for (0..steps_per_episode) |_| {
            const result = policy.evaluate(&state);
            const action = result.action;

            // Simulate environment: reward is higher when action matches
            // the quadrant of the state (simple toy problem)
            const quadrant = @as(usize, @intFromFloat(state[0] * @as(f32, @floatFromInt(ACTIONS)))) % ACTIONS;
            const reward: f32 = if (action == quadrant) 1.0 else -0.1;

            var next_state: [FEATURES]f32 = undefined;
            for (0..FEATURES) |i| next_state[i] = rng.float(f32);

            buffer.record(state, @intCast(action), reward, next_state, false, @intCast(ep));
            state = next_state;
        }
    }

    std.debug.print("  [2/5] Collected {d} transitions ({d} episodes x {d} steps)\n", .{
        buffer.count,
        episodes,
        steps_per_episode,
    });
    std.debug.print("        Avg reward: {d:.3}  Fill: {d:.0}%\n", .{
        buffer.avgReward(),
        buffer.fillRate() * 100,
    });

    // ── Step 3: Train from experience ────────────────────────────
    const train_steps = 5000;
    for (0..train_steps) |_| {
        const t = buffer.samplePrioritized(&rng);
        policy.update(&t.state, t.action, t.reward, 0.01);
    }

    std.debug.print("  [3/5] Trained for {d} steps (prioritized replay)\n", .{train_steps});

    // Evaluate accuracy after training
    var correct: usize = 0;
    const eval_trials = 1000;
    for (0..eval_trials) |_| {
        var state: [FEATURES]f32 = undefined;
        for (0..FEATURES) |i| state[i] = rng.float(f32);
        const result = policy.evaluate(&state);
        const quadrant = @as(usize, @intFromFloat(state[0] * @as(f32, @floatFromInt(ACTIONS)))) % ACTIONS;
        if (result.action == quadrant) correct += 1;
    }
    std.debug.print("        Accuracy: {d}/{d} ({d:.1}%)\n", .{
        correct,
        eval_trials,
        @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(eval_trials)) * 100,
    });

    // ── Step 4: Quantize to BitNet 1.58-bit ──────────────────────
    const quant = znap.quantized.TernaryPolicy(FEATURES, ACTIONS).fromFloat(&policy);

    std.debug.print("  [4/5] Quantized to BitNet b1.58\n", .{});
    std.debug.print("        Float:    {d} bytes\n", .{@TypeOf(policy).param_count * 4});
    std.debug.print("        Ternary:  {d} bytes ({d:.1}x compression)\n", .{
        znap.quantized.TernaryPolicy(FEATURES, ACTIONS).memoryBytes(),
        @as(f32, @floatFromInt(@TypeOf(policy).param_count * 4)) /
            @as(f32, @floatFromInt(znap.quantized.TernaryPolicy(FEATURES, ACTIONS).memoryBytes())),
    });

    // ── Step 5: Compare float vs quantized ───────────────────────
    var agreement: usize = 0;
    var quant_correct: usize = 0;
    for (0..eval_trials) |_| {
        var state: [FEATURES]f32 = undefined;
        for (0..FEATURES) |i| state[i] = rng.float(f32);
        const float_r = policy.evaluate(&state);
        const quant_r = quant.evaluate(&state);
        if (float_r.action == quant_r.action) agreement += 1;
        const quadrant = @as(usize, @intFromFloat(state[0] * @as(f32, @floatFromInt(ACTIONS)))) % ACTIONS;
        if (quant_r.action == quadrant) quant_correct += 1;
    }

    std.debug.print("  [5/5] Float vs Ternary agreement: {d:.1}%\n", .{
        @as(f32, @floatFromInt(agreement)) / @as(f32, @floatFromInt(eval_trials)) * 100,
    });
    std.debug.print("        Quantized accuracy: {d:.1}%\n", .{
        @as(f32, @floatFromInt(quant_correct)) / @as(f32, @floatFromInt(eval_trials)) * 100,
    });

    std.debug.print(
        \\
        \\  Done. Run `zig build bench` to see inference latency.
        \\
    , .{});
}
