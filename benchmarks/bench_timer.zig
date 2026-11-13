//! Monotonic timer for the experiment binaries (std.time.Timer was removed in Zig 0.16).
const std = @import("std");

/// Set once from `main(init)` before any Timer is used.
pub var io: std.Io = undefined;

pub const Timer = struct {
    t0: i96,

    pub fn start() error{}!Timer {
        return .{ .t0 = std.Io.Clock.awake.now(io).nanoseconds };
    }

    pub fn read(self: Timer) u64 {
        return @intCast(std.Io.Clock.awake.now(io).nanoseconds - self.t0);
    }
};
