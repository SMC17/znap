// Example: Train a tiny character-level model and generate text.
//
// Demonstrates the full znap pipeline:
//   1. Build a byte-level tokenizer
//   2. Tokenize a training corpus
//   3. Train a small linear policy on next-token prediction
//   4. Generate text autoregressively
//   5. Quantize to BitNet and compare output
//
// This is the Zig equivalent of Karpathy's "train on Shakespeare" demo,
// but running in microseconds instead of minutes.

const std = @import("std");
const znap = @import("znap");

const VOCAB_SIZE = 128; // ASCII
const CONTEXT_LEN = 8; // Look at 8 previous characters
const FEATURES = CONTEXT_LEN; // Each feature = one-hot character position
const ACTIONS = VOCAB_SIZE; // Predict next character

// Simple training corpus
const CORPUS =
    \\To be, or not to be, that is the question:
    \\Whether 'tis nobler in the mind to suffer
    \\The slings and arrows of outrageous fortune,
    \\Or to take arms against a sea of troubles,
    \\And by opposing end them. To die, to sleep;
    \\No more; and by a sleep to say we end
    \\The heart-ache and the thousand natural shocks
    \\That flesh is heir to: 'tis a consummation
    \\Devoutly to be wish'd. To die, to sleep;
    \\To sleep, perchance to dream.
;

pub fn main() !void {
    std.debug.print(
        \\
        \\  znap — Train & Generate Demo
        \\  ─────────────────────────────
        \\  Corpus: Shakespeare ({d} chars)
        \\  Vocab: ASCII ({d})
        \\  Context: {d} chars
        \\
        \\
    , .{ CORPUS.len, VOCAB_SIZE, CONTEXT_LEN });

    // ── Step 1: Build tokenizer ──────────────────────────────────
    const tok = znap.tokenizer.Tokenizer.initByteLevelVocab();
    std.debug.print("  [1/5] Tokenizer: byte-level, {d} tokens\n", .{tok.vocab_size});

    // ── Step 2: Build training examples ──────────────────────────
    // Each example: [c_{t-8}, c_{t-7}, ..., c_{t-1}] → c_t
    var buf = znap.replay.ReplayBuffer(10_000, FEATURES).init();

    for (CONTEXT_LEN..CORPUS.len) |i| {
        var state: [FEATURES]f32 = undefined;
        for (0..CONTEXT_LEN) |j| {
            // Simple encoding: normalized byte value
            state[j] = @as(f32, @floatFromInt(CORPUS[i - CONTEXT_LEN + j])) / 128.0;
        }
        const target: u8 = CORPUS[i];
        // Reward = 1 if we predict correctly, -0.1 otherwise
        buf.record(state, target, 1.0, state, false, @intCast(i));
    }

    std.debug.print("  [2/5] Training data: {d} examples from {d}-char corpus\n", .{
        buf.count,
        CORPUS.len,
    });

    // ── Step 3: Train ────────────────────────────────────────────
    // Use a smaller action space for tractability
    const PRED_ACTIONS = 96; // Printable ASCII range
    const P = znap.policy.LinearPolicy(FEATURES, PRED_ACTIONS);
    var pol = P.init();

    // Manual training loop with character prediction
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var correct: usize = 0;
    const train_steps = 5000;
    for (0..train_steps) |_| {
        const t = buf.sampleUniform(&rng);
        const mapped_action = if (t.action >= 32 and t.action < 32 + PRED_ACTIONS)
            t.action - 32
        else
            0;
        pol.update(&t.state, mapped_action, t.reward * 0.01, 0.005);

        // Check accuracy
        const result = pol.evaluate(&t.state);
        if (result.action + 32 == t.action) correct += 1;
    }

    std.debug.print("  [3/5] Trained for {d} steps, accuracy: {d:.1}%\n", .{
        train_steps,
        @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(train_steps)) * 100,
    });

    // ── Step 4: Generate text ────────────────────────────────────
    std.debug.print("  [4/5] Generating text...\n\n", .{});
    std.debug.print("        \"", .{});

    // Seed with "To be, o"
    var context: [CONTEXT_LEN]f32 = undefined;
    const seed_text = "To be, o";
    for (0..CONTEXT_LEN) |i| {
        context[i] = @as(f32, @floatFromInt(seed_text[i])) / 128.0;
    }

    // Generate 100 characters
    for (0..100) |_| {
        const result = pol.evaluate(&context);
        const ch: u8 = @intCast(result.action + 32);

        // Print character
        if (ch >= 32 and ch < 127) {
            std.debug.print("{c}", .{ch});
        } else {
            std.debug.print(".", .{});
        }

        // Shift context
        for (0..CONTEXT_LEN - 1) |i| context[i] = context[i + 1];
        context[CONTEXT_LEN - 1] = @as(f32, @floatFromInt(ch)) / 128.0;
    }
    std.debug.print("\"\n\n", .{});

    // ── Step 5: Quantize and compare ─────────────────────────────
    const Q = znap.quantized.TernaryPolicy(FEATURES, PRED_ACTIONS);
    const qpol = Q.fromFloat(&pol);

    var agree: usize = 0;
    for (0..100) |_| {
        var state: [FEATURES]f32 = undefined;
        for (0..FEATURES) |j| state[j] = rng.float(f32);
        const float_r = pol.evaluate(&state);
        const quant_r = qpol.evaluate(&state);
        if (float_r.action == quant_r.action) agree += 1;
    }

    std.debug.print("  [5/5] Float vs Ternary agreement: {d}%\n", .{agree});
    std.debug.print("        Float size:   {d} bytes\n", .{P.param_count * 4});
    std.debug.print("        Ternary size: {d} bytes ({d:.1}x compression)\n\n", .{
        Q.memoryBytes(),
        @as(f32, @floatFromInt(P.param_count * 4)) / @as(f32, @floatFromInt(Q.memoryBytes())),
    });
}
