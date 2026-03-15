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

    const audit_exe = b.addExecutable(.{
        .name = "znap-audit",
        .root_source_file = b.path("benchmarks/audit.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    audit_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(audit_exe);

    const run_audit = b.addRunArtifact(audit_exe);
    const audit_step = b.step("audit", "Run honest audit with trained weights");
    audit_step.dependOn(&run_audit.step);

    const scaling_exe = b.addExecutable(.{
        .name = "znap-scaling",
        .root_source_file = b.path("benchmarks/scaling.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    scaling_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(scaling_exe);

    const run_scaling = b.addRunArtifact(scaling_exe);
    const scaling_step = b.step("scaling", "Run scaling law experiment across 7 model sizes");
    scaling_step.dependOn(&run_scaling.step);

    const crossover_exe = b.addExecutable(.{
        .name = "znap-dqn-crossover",
        .root_source_file = b.path("benchmarks/dqn_crossover.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    crossover_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(crossover_exe);

    const run_crossover = b.addRunArtifact(crossover_exe);
    const crossover_step = b.step("dqn-crossover", "DQN ternary quantization crossover experiment");
    crossover_step.dependOn(&run_crossover.step);

    const qat_exe = b.addExecutable(.{
        .name = "znap-qat",
        .root_source_file = b.path("benchmarks/qat.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    qat_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(qat_exe);

    const run_qat = b.addRunArtifact(qat_exe);
    const qat_step = b.step("qat", "Quantization-aware training experiment");
    qat_step.dependOn(&run_qat.step);

    const proof_exe = b.addExecutable(.{
        .name = "znap-formal",
        .root_source_file = b.path("benchmarks/formal_proof.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    proof_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(proof_exe);

    const run_proof = b.addRunArtifact(proof_exe);
    const proof_step = b.step("formal", "Formal verification + INT8 + softmax experiments");
    proof_step.dependOn(&run_proof.step);

    const margin_exe = b.addExecutable(.{
        .name = "znap-margin",
        .root_source_file = b.path("benchmarks/margin_training.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    margin_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(margin_exe);

    const run_margin = b.addRunArtifact(margin_exe);
    const margin_step = b.step("margin", "Margin training → quantization robustness experiment");
    margin_step.dependOn(&run_margin.step);

    const deep_exe = b.addExecutable(.{
        .name = "znap-deep-scaling",
        .root_source_file = b.path("benchmarks/deep_scaling.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    deep_exe.root_module.addImport("znap", lib.root_module);
    b.installArtifact(deep_exe);

    const run_deep = b.addRunArtifact(deep_exe);
    const deep_step = b.step("deep-scaling", "Deep scaling: DQN ternary crossover from 100 to 500K params");
    deep_step.dependOn(&run_deep.step);

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

    const gen_example = b.addExecutable(.{
        .name = "generate",
        .root_source_file = b.path("examples/generate.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    gen_example.root_module.addImport("znap", lib.root_module);
    b.installArtifact(gen_example);

    const run_gen = b.addRunArtifact(gen_example);
    if (b.args) |args| {
        run_gen.addArgs(args);
    }
    const run_step = b.step("run", "Generate text from a pretrained model");
    run_step.dependOn(&run_gen.step);

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
        "src/mla.zig",
        "src/moe.zig",
        "src/speculative.zig",
        "src/gqa.zig",
        "src/ssm.zig",
        "src/weights.zig",
        "src/tokenizer.zig",
        "src/train.zig",
        "src/gpu.zig",
        "src/gpu_train.zig",
        "src/gguf.zig",
        "src/quant_matmul.zig",
        "src/int8_matmul.zig",
        "src/w8a8.zig",
        "src/neon.zig",
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
