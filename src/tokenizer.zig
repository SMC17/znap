// Byte-Pair Encoding (BPE) tokenizer.
//
// Minimal BPE implementation for text tokenization. Supports:
//   - Character-level encoding (no vocabulary needed)
//   - BPE merges from a merge table
//   - Encode text → token IDs
//   - Decode token IDs → text
//
// The vocabulary and merge rules are loaded at runtime (not comptime)
// since they depend on the training corpus.

const std = @import("std");

/// Maximum vocabulary size.
pub const MAX_VOCAB: usize = 4096;
/// Maximum token length in bytes.
pub const MAX_TOKEN_LEN: usize = 32;
/// Maximum merge rules.
pub const MAX_MERGES: usize = 4096;

/// A single BPE merge rule: pair (a, b) → merged token c.
pub const MergeRule = struct {
    a: u32,
    b: u32,
    merged: u32,
};

/// Byte-level BPE tokenizer.
pub const Tokenizer = struct {
    /// Vocabulary: token_id → bytes.
    vocab: [MAX_VOCAB][MAX_TOKEN_LEN]u8,
    vocab_lens: [MAX_VOCAB]u8,
    vocab_size: u32,

    /// Merge rules in priority order.
    merges: [MAX_MERGES]MergeRule,
    num_merges: u32,

    /// Special tokens.
    bos_token: u32,
    eos_token: u32,
    pad_token: u32,

    const Self = @This();

    /// Initialize with byte-level vocabulary (256 single-byte tokens).
    pub fn initByteLevelVocab() Self {
        var self: Self = undefined;
        self.vocab_size = 256;
        self.num_merges = 0;
        self.bos_token = 0;
        self.eos_token = 0;
        self.pad_token = 0;

        // Initialize byte-level vocab: token i = byte i
        for (0..256) |i| {
            self.vocab[i][0] = @intCast(i);
            self.vocab_lens[i] = 1;
        }
        // Zero out the rest
        for (256..MAX_VOCAB) |i| {
            self.vocab_lens[i] = 0;
        }

        return self;
    }

    /// Add a merge rule: when tokens `a` and `b` appear adjacent, replace with `merged`.
    pub fn addMerge(self: *Self, a: u32, b: u32) u32 {
        if (self.num_merges >= MAX_MERGES) return 0;
        if (self.vocab_size >= MAX_VOCAB) return 0;

        const merged = self.vocab_size;

        // The merged token's bytes = bytes(a) ++ bytes(b)
        const len_a = self.vocab_lens[a];
        const len_b = self.vocab_lens[b];
        const new_len = len_a + len_b;
        if (new_len > MAX_TOKEN_LEN) return 0;

        for (0..len_a) |i| {
            self.vocab[merged][i] = self.vocab[a][i];
        }
        for (0..len_b) |i| {
            self.vocab[merged][len_a + i] = self.vocab[b][i];
        }
        self.vocab_lens[merged] = @intCast(new_len);

        self.merges[self.num_merges] = .{ .a = a, .b = b, .merged = merged };
        self.num_merges += 1;
        self.vocab_size += 1;

        return merged;
    }

    /// Encode a byte string into token IDs.
    /// Uses a fixed-size buffer (no heap allocation for short inputs).
    pub fn encode(self: *const Self, text: []const u8, output: []u32) usize {
        if (text.len == 0) return 0;

        // Start with byte-level tokens
        var tokens: [4096]u32 = undefined;
        var n: usize = @min(text.len, 4096);
        for (0..n) |i| {
            tokens[i] = text[i];
        }

        // Apply merge rules in priority order
        for (0..self.num_merges) |m| {
            const rule = &self.merges[m];
            var i: usize = 0;
            while (i + 1 < n) {
                if (tokens[i] == rule.a and tokens[i + 1] == rule.b) {
                    tokens[i] = rule.merged;
                    // Shift remaining tokens left
                    var j = i + 1;
                    while (j + 1 < n) : (j += 1) {
                        tokens[j] = tokens[j + 1];
                    }
                    n -= 1;
                    // Don't advance i — check for consecutive merges
                } else {
                    i += 1;
                }
            }
        }

        // Copy to output
        const out_len = @min(n, output.len);
        for (0..out_len) |i| {
            output[i] = tokens[i];
        }
        return out_len;
    }

    /// Decode token IDs back to bytes.
    pub fn decode(self: *const Self, tokens: []const u32, output: []u8) usize {
        var pos: usize = 0;
        for (tokens) |token_id| {
            if (token_id >= self.vocab_size) continue;
            const len = self.vocab_lens[token_id];
            if (pos + len > output.len) break;
            for (0..len) |i| {
                output[pos + i] = self.vocab[token_id][i];
            }
            pos += len;
        }
        return pos;
    }

    /// Get the string representation of a token.
    pub fn tokenToStr(self: *const Self, token_id: u32) []const u8 {
        if (token_id >= self.vocab_size) return "";
        return self.vocab[token_id][0..self.vocab_lens[token_id]];
    }
};

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "byte-level vocab init" {
    const tok = Tokenizer.initByteLevelVocab();
    try std.testing.expectEqual(@as(u32, 256), tok.vocab_size);
    try std.testing.expectEqual(@as(u8, 1), tok.vocab_lens[0]);
    try std.testing.expectEqual(@as(u8, 1), tok.vocab_lens[255]);
    try std.testing.expectEqual(@as(u8, 65), tok.vocab[65][0]); // 'A'
}

test "encode without merges is byte-level" {
    const tok = Tokenizer.initByteLevelVocab();
    var output: [100]u32 = undefined;
    const n = tok.encode("hello", &output);
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqual(@as(u32, 'h'), output[0]);
    try std.testing.expectEqual(@as(u32, 'e'), output[1]);
    try std.testing.expectEqual(@as(u32, 'l'), output[2]);
    try std.testing.expectEqual(@as(u32, 'l'), output[3]);
    try std.testing.expectEqual(@as(u32, 'o'), output[4]);
}

test "decode round-trip" {
    const tok = Tokenizer.initByteLevelVocab();
    const text = "Hello, world!";
    var tokens: [100]u32 = undefined;
    const n = tok.encode(text, &tokens);

    var decoded: [100]u8 = undefined;
    const m = tok.decode(tokens[0..n], &decoded);

    try std.testing.expectEqualStrings(text, decoded[0..m]);
}

test "add merge and encode" {
    var tok = Tokenizer.initByteLevelVocab();

    // Merge 'h' + 'e' → 256
    const he = tok.addMerge('h', 'e');
    try std.testing.expectEqual(@as(u32, 256), he);
    try std.testing.expectEqual(@as(u32, 257), tok.vocab_size);

    var output: [100]u32 = undefined;
    const n = tok.encode("hello", &output);
    // 'h'+'e' merged → 256, then 'l', 'l', 'o'
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqual(@as(u32, 256), output[0]); // "he"
    try std.testing.expectEqual(@as(u32, 'l'), output[1]);
}

test "multiple merges" {
    var tok = Tokenizer.initByteLevelVocab();

    const he = tok.addMerge('h', 'e');
    const ll = tok.addMerge('l', 'l');

    var output: [100]u32 = undefined;
    const n = tok.encode("hello", &output);
    // "he" → 256, "ll" → 257, "o"
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(he, output[0]);
    try std.testing.expectEqual(ll, output[1]);
    try std.testing.expectEqual(@as(u32, 'o'), output[2]);
}

test "decode merged tokens" {
    var tok = Tokenizer.initByteLevelVocab();
    _ = tok.addMerge('h', 'e');
    _ = tok.addMerge('l', 'l');

    var tokens: [100]u32 = undefined;
    const n = tok.encode("hello", &tokens);

    var decoded: [100]u8 = undefined;
    const m = tok.decode(tokens[0..n], &decoded);
    try std.testing.expectEqualStrings("hello", decoded[0..m]);
}

test "tokenToStr" {
    var tok = Tokenizer.initByteLevelVocab();
    const he = tok.addMerge('h', 'e');

    const s = tok.tokenToStr(he);
    try std.testing.expectEqualStrings("he", s);
    try std.testing.expectEqualStrings("A", tok.tokenToStr('A'));
}

test "empty input" {
    const tok = Tokenizer.initByteLevelVocab();
    var output: [100]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 0), tok.encode("", &output));
}

test "hierarchical merges" {
    var tok = Tokenizer.initByteLevelVocab();
    const th = tok.addMerge('t', 'h');
    const the = tok.addMerge(th, 'e');

    var output: [100]u32 = undefined;
    const n = tok.encode("the", &output);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(the, output[0]);

    var decoded: [100]u8 = undefined;
    const m = tok.decode(output[0..n], &decoded);
    try std.testing.expectEqualStrings("the", decoded[0..m]);
}
