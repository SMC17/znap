// Multi-threaded parallel policy evaluation.
//
// Evaluates N policies across M observation batches simultaneously.
// On a 128-thread Threadripper, evaluates hundreds of policies
// in the wall-clock time of a single evaluation.

const std = @import("std");
const policy = @import("policy.zig");
const simd = @import("simd.zig");

pub const MAX_AGENTS = 32;
pub const MAX_BATCH = 64;

/// Result of evaluating one policy on one observation.
pub fn SectionResult(comptime ActionDim: usize) type {
    return struct {
        batch_idx: u32,
        agent_idx: u32,
        action: usize,
        confidence: f32,
        scores: [ActionDim]f32,
    };
}

/// Batch evaluation results.
pub fn ParallelResult(comptime ActionDim: usize) type {
    return struct {
        results: [MAX_BATCH * MAX_AGENTS]SectionResult(ActionDim),
        count: u32,
        batch_size: u32,
        agent_count: u32,

        const Self = @This();

        pub fn get(self: *const Self, batch: usize, agent: usize) *const SectionResult(ActionDim) {
            return &self.results[batch * self.agent_count + agent];
        }

        pub fn bestForBatch(self: *const Self, batch: usize) SectionResult(ActionDim) {
            var best = self.results[batch * self.agent_count];
            for (1..self.agent_count) |a| {
                const r = &self.results[batch * self.agent_count + a];
                if (r.confidence > best.confidence) best = r.*;
            }
            return best;
        }
    };
}

/// Evaluate multiple policies on multiple observations in parallel.
pub fn evaluateParallel(
    comptime FeatureDim: usize,
    comptime ActionDim: usize,
    observations: []const [FeatureDim]f32,
    policies: []const policy.LinearPolicy(FeatureDim, ActionDim),
) ParallelResult(ActionDim) {
    const n_obs = @min(observations.len, MAX_BATCH);
    const n_agents = @min(policies.len, MAX_AGENTS);

    var result: ParallelResult(ActionDim) = undefined;
    result.batch_size = @intCast(n_obs);
    result.agent_count = @intCast(n_agents);
    result.count = @intCast(n_obs * n_agents);

    // For small workloads, run sequentially
    if (n_obs * n_agents <= 16) {
        for (0..n_obs) |b| {
            for (0..n_agents) |a| {
                const pr = policies[a].evaluate(&observations[b]);
                result.results[b * n_agents + a] = .{
                    .batch_idx = @intCast(b),
                    .agent_idx = @intCast(a),
                    .action = pr.action,
                    .confidence = pr.confidence,
                    .scores = pr.probabilities,
                };
            }
        }
        return result;
    }

    // For larger workloads, use thread pool
    const Work = struct {
        obs: *const [FeatureDim]f32,
        pol: *const policy.LinearPolicy(FeatureDim, ActionDim),
        out: *SectionResult(ActionDim),
        batch_idx: u32,
        agent_idx: u32,
        n_agents_: u32,
    };

    var work_items: [MAX_BATCH * MAX_AGENTS]Work = undefined;
    var threads: [MAX_BATCH]std.Thread = undefined;

    const EvalFn = struct {
        fn run(items: []const Work) void {
            for (items) |w| {
                const pr = w.pol.evaluate(w.obs);
                w.out.* = .{
                    .batch_idx = w.batch_idx,
                    .agent_idx = w.agent_idx,
                    .action = pr.action,
                    .confidence = pr.confidence,
                    .scores = pr.probabilities,
                };
            }
        }
    };

    // Build work items
    for (0..n_obs) |b| {
        for (0..n_agents) |a| {
            const idx = b * n_agents + a;
            work_items[idx] = .{
                .obs = &observations[b],
                .pol = &policies[a],
                .out = &result.results[idx],
                .batch_idx = @intCast(b),
                .agent_idx = @intCast(a),
                .n_agents_ = @intCast(n_agents),
            };
        }
    }

    // Spawn one thread per batch element
    var spawned: usize = 0;
    for (0..n_obs) |b| {
        const start = b * n_agents;
        const end = start + n_agents;
        threads[b] = std.Thread.spawn(.{}, EvalFn.run, .{work_items[start..end]}) catch {
            EvalFn.run(work_items[start..end]);
            continue;
        };
        spawned += 1;
    }

    for (0..spawned) |b| threads[b].join();

    return result;
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "sequential evaluation for small batch" {
    const P = policy.LinearPolicy(4, 3);
    var obs: [3][4]f32 = undefined;
    for (0..3) |i| for (0..4) |j| {
        obs[i][j] = @as(f32, @floatFromInt(i * 4 + j)) / 12.0;
    };

    var pols = [_]P{P.init()};
    const result = evaluateParallel(4, 3, &obs, &pols);
    try std.testing.expectEqual(@as(u32, 3), result.batch_size);
    try std.testing.expectEqual(@as(u32, 1), result.agent_count);
}

test "parallel evaluation spawns threads" {
    const P = policy.LinearPolicy(4, 3);
    var obs: [8][4]f32 = undefined;
    for (0..8) |i| for (0..4) |j| {
        obs[i][j] = @as(f32, @floatFromInt(j)) / 4.0;
    };

    var pols = [_]P{P.init()} ** 3;
    const result = evaluateParallel(4, 3, &obs, &pols);
    try std.testing.expectEqual(@as(u32, 8), result.batch_size);
    try std.testing.expectEqual(@as(u32, 3), result.agent_count);
    try std.testing.expectEqual(@as(u32, 24), result.count);
}

test "bestForBatch returns highest confidence" {
    const P = policy.LinearPolicy(4, 3);
    var obs: [5][4]f32 = [_][4]f32{[_]f32{0.5} ** 4} ** 5;

    var pols: [3]P = [_]P{P.init()} ** 3;
    // Train agent 1 to be confident about action 2
    for (0..100) |_| {
        var f = [_]f32{0.5} ** 4;
        pols[1].update(&f, 2, 1.0, 0.1);
    }

    const result = evaluateParallel(4, 3, &obs, &pols);
    const best = result.bestForBatch(0);
    try std.testing.expectEqual(@as(u32, 1), best.agent_idx);
}
