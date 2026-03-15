// Zero-allocation experience replay buffer.
//
// Fixed-size ring buffer with comptime-known dimensions.
// Entire buffer lives on the stack or in a contiguous arena.
// O(1) insert, O(1) random sample, priority-weighted sampling.

const std = @import("std");

/// A single experience transition: (s, a, r, s', done).
pub fn Transition(comptime StateDim: usize) type {
    return struct {
        state: [StateDim]f32,
        action: u8,
        reward: f32,
        next_state: [StateDim]f32,
        done: bool,
        priority: f32,
        tick: u64,

        const Self = @This();

        pub fn init() Self {
            return .{
                .state = [_]f32{0} ** StateDim,
                .action = 0,
                .reward = 0,
                .next_state = [_]f32{0} ** StateDim,
                .done = false,
                .priority = 1.0,
                .tick = 0,
            };
        }
    };
}

/// Fixed-capacity ring buffer for experience replay.
pub fn ReplayBuffer(comptime Capacity: usize, comptime StateDim: usize) type {
    const T = Transition(StateDim);

    return struct {
        transitions: [Capacity]T,
        head: usize,
        count: usize,
        priority_sum: f64,
        total_inserted: u64,

        const Self = @This();
        pub const capacity = Capacity;
        pub const state_dim = StateDim;
        pub const buffer_size_bytes = Capacity * @sizeOf(T);

        pub fn init() Self {
            return .{
                .transitions = [_]T{T.init()} ** Capacity,
                .head = 0,
                .count = 0,
                .priority_sum = 0,
                .total_inserted = 0,
            };
        }

        pub fn push(self: *Self, t: T) void {
            if (self.count == Capacity) {
                self.priority_sum -= self.transitions[self.head].priority;
            }
            self.transitions[self.head] = t;
            self.priority_sum += t.priority;
            self.head = (self.head + 1) % Capacity;
            if (self.count < Capacity) self.count += 1;
            self.total_inserted += 1;
        }

        pub fn record(
            self: *Self,
            state: [StateDim]f32,
            action: u8,
            reward: f32,
            next_state: [StateDim]f32,
            done: bool,
            tick: u64,
        ) void {
            self.push(.{
                .state = state,
                .action = action,
                .reward = reward,
                .next_state = next_state,
                .done = done,
                .priority = @max(@abs(reward) + 0.01, 0.01),
                .tick = tick,
            });
        }

        pub fn get(self: *const Self, idx: usize) *const T {
            if (idx >= self.count) @panic("index out of bounds");
            const real = if (self.count < Capacity) idx else (self.head + idx) % Capacity;
            return &self.transitions[real];
        }

        pub fn sampleUniform(self: *const Self, rng: *std.Random) *const T {
            if (self.count == 0) @panic("cannot sample from empty buffer");
            return self.get(rng.uintLessThan(usize, self.count));
        }

        pub fn sampleBatch(self: *const Self, comptime N: usize, rng: *std.Random) [N]*const T {
            if (self.count < N) @panic("not enough transitions");
            var batch: [N]*const T = undefined;
            for (0..N) |i| batch[i] = self.sampleUniform(rng);
            return batch;
        }

        pub fn samplePrioritized(self: *const Self, rng: *std.Random) *const T {
            if (self.count == 0) @panic("cannot sample from empty buffer");
            if (self.priority_sum <= 0) return self.sampleUniform(rng);
            var target = rng.float(f64) * self.priority_sum;
            for (0..self.count) |i| {
                const t = self.get(i);
                target -= t.priority;
                if (target <= 0) return t;
            }
            return self.get(self.count - 1);
        }

        pub fn avgReward(self: *const Self) f32 {
            if (self.count == 0) return 0;
            var sum: f32 = 0;
            for (0..self.count) |i| sum += self.get(i).reward;
            return sum / @as(f32, @floatFromInt(self.count));
        }

        pub fn isFull(self: *const Self) bool {
            return self.count == Capacity;
        }

        pub fn fillRate(self: *const Self) f32 {
            return @as(f32, @floatFromInt(self.count)) / @as(f32, @floatFromInt(Capacity));
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "push and get" {
    var buf = ReplayBuffer(5, 4).init();
    var t = Transition(4).init();
    t.state[0] = 1.0;
    t.reward = 0.5;
    buf.push(t);
    try std.testing.expectEqual(@as(usize, 1), buf.count);
    try std.testing.expectEqual(@as(f32, 1.0), buf.get(0).state[0]);
}

test "ring buffer wraps" {
    var buf = ReplayBuffer(3, 2).init();
    for (0..5) |i| {
        var t = Transition(2).init();
        t.reward = @as(f32, @floatFromInt(i));
        buf.push(t);
    }
    try std.testing.expectEqual(@as(usize, 3), buf.count);
    try std.testing.expect(buf.isFull());
    try std.testing.expectEqual(@as(f32, 2.0), buf.get(0).reward);
}

test "uniform sampling" {
    var buf = ReplayBuffer(100, 4).init();
    for (0..50) |i| {
        var t = Transition(4).init();
        t.reward = @as(f32, @floatFromInt(i));
        buf.push(t);
    }
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    for (0..20) |_| {
        const s = buf.sampleUniform(&rng);
        try std.testing.expect(s.reward >= 0 and s.reward < 50);
    }
}

test "prioritized sampling favors high priority" {
    var buf = ReplayBuffer(100, 4).init();
    for (0..10) |_| {
        var t = Transition(4).init();
        t.priority = 0.01;
        buf.push(t);
    }
    var high = Transition(4).init();
    high.priority = 100.0;
    high.reward = 999.0;
    buf.push(high);

    var prng = std.Random.DefaultPrng.init(12345);
    var rng = prng.random();
    var count: usize = 0;
    for (0..20) |_| {
        if (buf.samplePrioritized(&rng).reward == 999.0) count += 1;
    }
    try std.testing.expect(count >= 15);
}

test "buffer size fits in L3 cache" {
    // 100K transitions x 8-dim state ≈ reasonable size
    const size_kb = ReplayBuffer(100_000, 8).buffer_size_bytes / 1024;
    try std.testing.expect(size_kb < 50_000); // < 50MB
}
