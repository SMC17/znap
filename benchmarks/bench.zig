// znap inference benchmarks.
//
// Measures wall-clock latency for every operation in the library.
// Run: zig build bench

const std = @import("std");
const znap = @import("znap");

const F = 30; // Feature dimension
const H = 64; // Hidden dimension
const A = 6;  // Action dimension

const features: [F]f32 = blk: {
    var f: [F]f32 = undefined;
    for (0..F) |i| f[i] = @as(f32, @floatFromInt(i)) / @as(f32, F);
    break :blk f;
};
const ones: [F]f32 = [_]f32{1.0} ** F;

const bandit = znap.policy.LinearPolicy(F, A).init();
const dqn = znap.policy.DQNPolicy(F, H, A).initXavier(42);
const ensemble = znap.policy.Ensemble(7, F, A).init();
const ternary_bandit = znap.quantized.TernaryPolicy(F, A).init();
const ternary_dqn = znap.quantized.TernaryDQN(F, H, A).init();

fn bench(comptime name: []const u8, comptime iters: usize, comptime func: fn () callconv(.Unspecified) void) void {
    for (0..iters / 10) |_| func(); // warmup
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iters) |_| func();
    const ns = timer.read();
    const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const ops_per_sec = @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(ns)) / 1e9);
    std.debug.print("  {s:<45} {d:>8.3} us  {d:>12.0} ops/s\n", .{ name, us, ops_per_sec });
}

fn benchDot() void { std.mem.doNotOptimizeAway(znap.simd.vdot(F, &features, &ones)); }
fn benchSoftmax() void { std.mem.doNotOptimizeAway(znap.simd.vsoftmax(A, &[A]f32{ 1, 2, 3, 4, 5, 6 })); }
fn benchRelu() void { std.mem.doNotOptimizeAway(znap.simd.vrelu(H, &[_]f32{-0.5} ** H)); }
fn benchBandit() void { std.mem.doNotOptimizeAway(bandit.evaluate(&features)); }
fn benchDQN() void { std.mem.doNotOptimizeAway(dqn.evaluate(&features)); }
fn benchEnsemble() void { std.mem.doNotOptimizeAway(ensemble.evaluateAll(&features)); }
fn benchTernaryBandit() void { std.mem.doNotOptimizeAway(ternary_bandit.evaluate(&features)); }
fn benchTernaryDQN() void { std.mem.doNotOptimizeAway(ternary_dqn.evaluate(&features)); }

pub fn main() !void {
    const arch = @tagName(@import("builtin").cpu.arch);
    const os = @tagName(@import("builtin").os.tag);

    std.debug.print(
        \\
        \\  ╔══════════════════════════════════════════════════════════════╗
        \\  ║  znap — inference benchmarks                               ║
        \\  ║  Platform: {s} / {s}                              ║
        \\  ╚══════════════════════════════════════════════════════════════╝
        \\
        \\
    , .{ arch, os });

    std.debug.print("SIMD primitives ({d}-dim features):\n", .{F});
    bench("dot product", 10_000_000, benchDot);
    bench("softmax", 10_000_000, benchSoftmax);
    bench("relu", 10_000_000, benchRelu);

    std.debug.print("\nFloat32 policy evaluation:\n", .{});
    bench("linear 30x6 (bandit)", 1_000_000, benchBandit);
    bench("DQN 30x64x6 (2-layer)", 1_000_000, benchDQN);
    bench("ensemble 7x linear", 100_000, benchEnsemble);

    std.debug.print("\nBitNet b1.58 quantized (integer-only inner loop):\n", .{});
    bench("ternary linear 30x6", 1_000_000, benchTernaryBandit);
    bench("ternary DQN 30x64x6", 1_000_000, benchTernaryDQN);

    const bandit_mem = znap.policy.LinearPolicy(F, A).param_count * 4;
    const ternary_mem = znap.quantized.TernaryPolicy(F, A).memoryBytes();
    const dqn_mem = znap.policy.DQNPolicy(F, H, A).param_count * 4;

    std.debug.print(
        \\
        \\  ────────────────────────────────────────────────────────────
        \\  Memory footprint:
        \\    Linear (f32):  {d} bytes ({d} params)
        \\    Linear (1.58b): {d} bytes ({d:.1}x compression)
        \\    DQN (f32):     {d} bytes ({d} params)
        \\  ────────────────────────────────────────────────────────────
        \\  Zero heap allocations. All stack. All comptime shapes.
        \\  ────────────────────────────────────────────────────────────
        \\
    , .{
        bandit_mem,
        znap.policy.LinearPolicy(F, A).param_count,
        ternary_mem,
        @as(f32, @floatFromInt(bandit_mem)) / @as(f32, @floatFromInt(ternary_mem)),
        dqn_mem,
        znap.policy.DQNPolicy(F, H, A).param_count,
    });
}
