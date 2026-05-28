const std = @import("std");

// 31ns Micro-Prediction Challenge (standalone, modern-Zig compatible)
// Target: Linear / Ternary policy (30 features → 6 actions) evaluation.
// Reference bar (Apple M1 NEON SDOT, Zig ~0.14 ReleaseFast): 31 ns / 33 M evals/s.
// This file demonstrates the comptime + zero-alloc pattern on your hardware.
// Scalar fallback on x86_64; NEON fast path on aarch64 with the full src/.
//
// Build: zig build-exe 31ns-micro-challenge.zig -O ReleaseFast
// Run: ./31ns-micro-challenge
//
// Falsifier (pre-registered stax-experiment znap:31ns-m1-linear):
//   On equivalent Apple M1-class hardware + Zig 0.14 ReleaseFast + NEON path,
//   median time per eval > 62 ns across 1M iterations (or action selection
//   mismatches reference in >0.1% of cases) → claim refuted.
//
// Cross-lang harness: see 31ns-micro-challenge.py for NumPy baseline.
//
// This is sovereign micro-prediction ammo under Quantitative Mercantilism.
// Embed it. Time it. Beat it on your silicon. Then tell the story with receipts.

const F: usize = 30;
const A: usize = 6;

const TernaryPolicy = struct {
    // Packed 2-bit ternary + per-row f32 scale (matches quantized.zig layout)
    buf: [((F * A + 3) / 4) + A * 4]u8,

    fn getScale(self: *const @This(), row: usize) f32 {
        const off = (F * A + 3) / 4 + row * 4;
        return @bitCast(self.buf[off..][0..4].*);
    }

    fn get(self: *const @This(), row: usize, col: usize) i2 {
        const idx = row * F + col;
        const bo: u3 = @intCast((idx % 4) * 2);
        const bits: u2 = @truncate(self.buf[idx / 4] >> bo);
        return switch (bits) { 0b00 => 0, 0b01 => 1, 0b10 => -1, 0b11 => 0 };
    }
};

fn ternaryMatVec(comptime Rows: usize, comptime Cols: usize, policy: *const TernaryPolicy, x: *const [Cols]f32, out: *[Rows]f32) void {
    for (0..Rows) |r| {
        var acc: i32 = 0;
        for (0..Cols) |c| {
            const w = @as(i32, @intFromEnum(policy.get(r, c)));
            const xi: i32 = @intFromFloat(x[c] * 127.0); // crude dequant for demo; real path uses Q8_0
            acc += w * xi;
        }
        const scale = policy.getScale(r);
        out[r] = @as(f32, @floatFromInt(acc)) * scale;
    }
}

fn argmax(comptime N: usize, scores: *const [N]f32) usize {
    var best: usize = 0;
    var best_val = scores[0];
    for (1..N) |i| {
        if (scores[i] > best_val) {
            best_val = scores[i];
            best = i;
        }
    }
    return best;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    // Synthetic tiny ternary policy (30x6). In real use: load from .znap or embed trained weights.
    var policy: TernaryPolicy = undefined;
    @memset(&policy.buf, 0);
    // Set a few non-zero weights for non-trivial eval (demo only)
    // Real challenge would embed actual trained ternary weights from the paper experiments.
    for (0..A) |r| {
        const scale: f32 = 0.01;
        const off = (F * A + 3) / 4 + r * 4;
        policy.buf[off..][0..4].* = @bitCast(scale);
        // sprinkle a few +1/-1
        if (r < F) policy.buf[(r * F + r) / 4] |= (@as(u8, 0b01) << @intCast(((r * F + r) % 4) * 2));
    }

    var features: [F]f32 = undefined;
    for (0..F) |i| features[i] = @as(f32, @floatFromInt(i)) / @as(f32, F);

    var scores: [A]f32 = undefined;

    // Warmup
    for (0..10_000) |_| {
        ternaryMatVec(A, F, &policy, &features, &scores);
        _ = argmax(A, &scores);
    }

    // Timed loop — 1M iterations for stable measurement
    const iters: usize = 1_000_000;
    var timer = try std.time.Timer.start();
    var sum_actions: u64 = 0;
    for (0..iters) |_| {
        ternaryMatVec(A, F, &policy, &features, &scores);
        const act = argmax(A, &scores);
        sum_actions +%= act;
    }
    const elapsed_ns = timer.read();
    const per_eval_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters));
    const evals_per_sec = @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    std.debug.print("\n=== 31ns Micro-Prediction Challenge ===\n", .{});
    std.debug.print("Platform: {s} {s}\n", .{ @tagName(@import("builtin").cpu.arch), @tagName(@import("builtin").os.tag) });
    std.debug.print("Iters: {d}\n", .{iters});
    std.debug.print("Total: {d:.3} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e6});
    std.debug.print("Per eval: {d:.1} ns\n", .{per_eval_ns});
    std.debug.print("Evals/s : {d:.1} M\n", .{evals_per_sec / 1e6});
    std.debug.print("Sanity (sum actions mod 6): {d}\n", .{sum_actions % 6});
    std.debug.print("\nReference bar (Apple M1 NEON SDOT, Zig ~0.14 ReleaseFast): 31 ns / 33 M evals/s\n", .{});
    std.debug.print("Falsifier: >62 ns median on equivalent M1-class hardware → claim refuted.\n", .{});
    std.debug.print("See AUDIT_NOTES/ and the full znap src/ for the real NEON SDOT + comptime kernels.\n", .{});
    std.debug.print("QM army-building note: Beat this on your silicon with zero Python rent.\n\n", .{});
}
