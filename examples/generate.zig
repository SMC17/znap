// Generate text from a pretrained model.
//
// Usage:
//   zig build run -- models/stories15M.bin
//   zig build run -- models/stories15M.bin -t 0.8 -n 256
//
// Downloads models from: https://huggingface.co/karpathy/tinyllamas

const std = @import("std");
const llama2 = @import("znap").llama2;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse args
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_path: []const u8 = "models/stories15M.bin";
    var tokenizer_path: []const u8 = "models/tokenizer.bin";
    var temperature: f32 = 0.9;
    var max_tokens: u32 = 256;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            i += 1;
            temperature = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "-n") and i + 1 < args.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "-k") and i + 1 < args.len) {
            i += 1;
            tokenizer_path = args[i];
        } else if (args[i][0] != '-') {
            model_path = args[i];
        }
    }

    // Load model
    const stderr = std.io.getStdErr().writer();
    try stderr.print("\n  znap — loading {s}...\n", .{model_path});

    var model = try llama2.Model.load(model_path, allocator);
    defer model.deinit(allocator);

    const c = model.config;
    const abs_vocab: u32 = @intCast(if (c.vocab_size < 0) -c.vocab_size else c.vocab_size);

    try stderr.print("  Config: dim={d} hidden={d} layers={d} heads={d} vocab={d} seq={d}\n", .{
        c.dim, c.hidden_dim, c.n_layers, c.n_heads, abs_vocab, c.seq_len,
    });
    try stderr.print("  Params: ~{d}M\n", .{
        (c.dim * abs_vocab + c.n_layers * (4 * c.dim * c.dim + 3 * c.hidden_dim * c.dim + 2 * c.dim)) / 1_000_000,
    });
    try stderr.print("  Temperature: {d:.1}\n", .{temperature});
    try stderr.print("  Max tokens: {d}\n\n", .{max_tokens});

    // Load tokenizer
    var tokenizer_inst = try llama2.Tokenizer.load(tokenizer_path, abs_vocab, allocator);
    defer tokenizer_inst.deinit(allocator);

    // Allocate run state
    var state = try llama2.RunState.init(model.config, allocator);
    defer state.deinit(allocator);

    // Generate
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    var rng = prng.random();

    const stdout = std.io.getStdOut().writer();

    var timer = try std.time.Timer.start();

    try llama2.generate(&model, &tokenizer_inst, &state, "", max_tokens, temperature, &rng, stdout);

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1e6;
    const tokens_per_sec = @as(f64, @floatFromInt(max_tokens)) / (elapsed_ms / 1000.0);

    try stdout.writeAll("\n");
    try stderr.print("\n  ---\n", .{});
    try stderr.print("  {d} tokens in {d:.0}ms ({d:.1} tok/s)\n\n", .{
        max_tokens, elapsed_ms, tokens_per_sec,
    });
}
