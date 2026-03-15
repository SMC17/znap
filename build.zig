const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Static library (C ABI) ───────────────────────────────────
    const lib = b.addStaticLibrary(.{
        .name = "znap",
        .root_source_file = b.path("src/znap.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // ── Shared library ───────────────────────────────────────────
    const shared = b.addSharedLibrary(.{
        .name = "znap",
        .root_source_file = b.path("src/znap.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(shared);

    // ── Benchmark runner ─────────────────────────────────────────
    const bench_exe = b.addExecutable(.{
        .name = "znap-bench",
        .root_source_file = b.path("benchmarks/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(bench_exe);

    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run inference benchmarks");
    bench_step.dependOn(&run_bench.step);

    // ── Examples ─────────────────────────────────────────────────
    const rl_example = b.addExecutable(.{
        .name = "rl-agent",
        .root_source_file = b.path("examples/rl_agent.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    rl_example.root_module.addImport("znap", lib.root_module);
    b.installArtifact(rl_example);

    const run_example = b.addRunArtifact(rl_example);
    const example_step = b.step("example", "Run the RL agent example");
    example_step.dependOn(&run_example.step);

    const train_example = b.addExecutable(.{
        .name = "train-and-generate",
        .root_source_file = b.path("examples/train_and_generate.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    train_example.root_module.addImport("znap", lib.root_module);
    b.installArtifact(train_example);

    const run_train = b.addRunArtifact(train_example);
    const train_step = b.step("train", "Train a tiny model and generate text");
    train_step.dependOn(&run_train.step);

    // ── Tests ────────────────────────────────────────────────────
    const test_files = [_][]const u8{
        "src/tensor.zig",
        "src/simd.zig",
        "src/quantized.zig",
        "src/policy.zig",
        "src/replay.zig",
        "src/parallel.zig",
        "src/attention.zig",
        "src/transformer.zig",
        "src/weights.zig",
        "src/tokenizer.zig",
        "src/train.zig",
        "src/znap.zig",
    };

    const test_step = b.step("test", "Run all unit tests");

    for (test_files) |file| {
        const t = b.addTest(.{
            .root_source_file = b.path(file),
            .target = target,
            .optimize = optimize,
        });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
