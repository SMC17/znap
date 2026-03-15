// Weight I/O for znap models.
//
// Loads and saves model weights using a simple binary format (.znap).
// This module is NOT on the inference hot path — allocations are fine here.
// The API is designed to be dead simple to use from both Zig and Python.
//
// .znap binary format:
//   Header:
//     magic:        [4]u8  = "ZNAP"
//     version:      u32    = 1
//     num_tensors:  u32
//     metadata_len: u32    (reserved, currently 0)
//
//   Per tensor entry:
//     name_len: u16
//     name:     [name_len]u8
//     ndim:     u8
//     dims:     [ndim]u32
//     dtype:    u8  (0 = f32, 1 = f16, 2 = bf16)
//     data:     [product(dims) * sizeof(dtype)]u8
//
// Python export example (see doc comment on writeHeader).

const std = @import("std");
const tensor = @import("tensor.zig");
const policy = @import("policy.zig");
const transformer_mod = @import("transformer.zig");
const attention = @import("attention.zig");

// ── Constants ────────────────────────────────────────────────────

pub const MAGIC = [4]u8{ 'Z', 'N', 'A', 'P' };
pub const FORMAT_VERSION: u32 = 1;

pub const DType = enum(u8) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
};

pub const WeightError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedDtype,
    ShapeMismatch,
    TensorNotFound,
    UnexpectedEof,
    TooManyDims,
};

// ── Header ───────────────────────────────────────────────────────

pub const Header = struct {
    version: u32,
    num_tensors: u32,
    metadata_len: u32,
};

/// TensorEntry describes a single named tensor in a .znap file.
pub const TensorEntry = struct {
    name: []const u8,
    ndim: u8,
    dims: [8]u32, // max 8 dimensions
    dtype: DType,
    data_offset: u64, // byte offset of data in the file stream
    data_len: u64, // byte length of tensor data
};

// ── Reading ──────────────────────────────────────────────────────

/// Read the .znap header. Returns error if magic/version mismatch.
pub fn readHeader(reader: anytype) !Header {
    var magic: [4]u8 = undefined;
    const bytes_read = try reader.readAll(&magic);
    if (bytes_read < 4) return WeightError.UnexpectedEof;
    if (!std.mem.eql(u8, &magic, &MAGIC)) return WeightError.InvalidMagic;

    const version = try reader.readInt(u32, .little);
    if (version != FORMAT_VERSION) return WeightError.UnsupportedVersion;

    const num_tensors = try reader.readInt(u32, .little);
    const metadata_len = try reader.readInt(u32, .little);

    // Skip metadata for now
    if (metadata_len > 0) {
        try reader.skipBytes(metadata_len, .{});
    }

    return .{
        .version = version,
        .num_tensors = num_tensors,
        .metadata_len = metadata_len,
    };
}

/// Read a tensor entry header (name, shape, dtype) but not the data bytes.
/// Returns the entry and leaves the reader positioned at the start of data.
pub fn readTensorMeta(reader: anytype, allocator: std.mem.Allocator) !TensorEntry {
    const name_len = try reader.readInt(u16, .little);
    const name = try allocator.alloc(u8, name_len);
    const n_read = try reader.readAll(name);
    if (n_read < name_len) return WeightError.UnexpectedEof;

    const ndim = try reader.readByte();
    if (ndim > 8) return WeightError.TooManyDims;

    var dims: [8]u32 = [_]u32{0} ** 8;
    var numel: u64 = 1;
    for (0..ndim) |i| {
        dims[i] = try reader.readInt(u32, .little);
        numel *= dims[i];
    }

    const dtype_byte = try reader.readByte();
    const dtype: DType = std.meta.intToEnum(DType, dtype_byte) catch return WeightError.UnsupportedDtype;

    const elem_size: u64 = switch (dtype) {
        .f32 => 4,
        .f16, .bf16 => 2,
    };

    return .{
        .name = name,
        .ndim = ndim,
        .dims = dims,
        .dtype = dtype,
        .data_offset = 0, // caller manages position
        .data_len = numel * elem_size,
    };
}

/// Read raw f32 data from the stream into a flat slice. Handles only f32 dtype.
pub fn readF32Data(reader: anytype, out: []f32) !void {
    const byte_slice = std.mem.sliceAsBytes(out);
    const n = try reader.readAll(byte_slice);
    if (n < byte_slice.len) return WeightError.UnexpectedEof;

    // Convert from little-endian if needed
    if (comptime @import("builtin").cpu.arch.endian() != .little) {
        for (out) |*v| {
            const raw: *[4]u8 = @ptrCast(v);
            std.mem.reverse(u8, raw);
        }
    }
}

/// Skip over a tensor's data bytes.
pub fn skipTensorData(reader: anytype, entry: *const TensorEntry) !void {
    try reader.skipBytes(entry.data_len, .{});
}

/// Find a named tensor in the stream. Reads entries sequentially.
/// Returns the entry positioned at data start. Skips non-matching entries.
fn findTensor(reader: anytype, allocator: std.mem.Allocator, name: []const u8, num_tensors: u32) !TensorEntry {
    for (0..num_tensors) |_| {
        var entry = try readTensorMeta(reader, allocator);
        if (std.mem.eql(u8, entry.name, name)) {
            return entry;
        }
        // Not the one we want — skip its data
        allocator.free(entry.name);
        try skipTensorData(reader, &entry);
    }
    return WeightError.TensorNotFound;
}

// ── Tensor from slice ────────────────────────────────────────────

/// Populate a comptime-sized Tensor from a flat f32 slice.
/// The slice must have exactly Rows * Cols elements.
pub fn loadTensorFromSlice(comptime Rows: usize, comptime Cols: usize, data: []const f32) !tensor.Tensor(Rows, Cols) {
    if (data.len != Rows * Cols) return WeightError.ShapeMismatch;
    var result: tensor.Tensor(Rows, Cols) = undefined;
    for (0..Rows) |r| {
        for (0..Cols) |c| {
            result.data[r][c] = data[r * Cols + c];
        }
    }
    return result;
}

/// Populate a 1D array from a flat f32 slice.
fn loadVecFromSlice(comptime N: usize, data: []const f32) ![N]f32 {
    if (data.len != N) return WeightError.ShapeMismatch;
    var result: [N]f32 = undefined;
    @memcpy(&result, data);
    return result;
}

// ── Read a named tensor into a Tensor(R,C) ───────────────────────

/// Read a named 2D tensor from an open reader. The reader must be positioned
/// after the header, and we scan sequentially through entries.
fn readNamedTensor(
    comptime Rows: usize,
    comptime Cols: usize,
    reader: anytype,
    allocator: std.mem.Allocator,
    name: []const u8,
    remaining: *u32,
) !tensor.Tensor(Rows, Cols) {
    while (remaining.* > 0) {
        remaining.* -= 1;
        var entry = try readTensorMeta(reader, allocator);
        defer allocator.free(entry.name);

        if (std.mem.eql(u8, entry.name, name)) {
            // Verify shape
            const expected_numel = Rows * Cols;
            var actual_numel: u64 = 1;
            for (0..entry.ndim) |i| actual_numel *= entry.dims[i];
            if (actual_numel != expected_numel) return WeightError.ShapeMismatch;
            if (entry.dtype != .f32) return WeightError.UnsupportedDtype;

            var buf: [Rows * Cols]f32 = undefined;
            try readF32Data(reader, &buf);
            return loadTensorFromSlice(Rows, Cols, &buf) catch unreachable;
        }
        try skipTensorData(reader, &entry);
    }
    return WeightError.TensorNotFound;
}

/// Read a named 1D vector (stored as Vec = Tensor(1, N)).
fn readNamedVec(
    comptime N: usize,
    reader: anytype,
    allocator: std.mem.Allocator,
    name: []const u8,
    remaining: *u32,
) !tensor.Vec(N) {
    const t = try readNamedTensor(1, N, reader, allocator, name, remaining);
    return t;
}

// ── File-based sequential loader ─────────────────────────────────
// Reads tensors in file order, matching by name. Each tensor is read
// exactly once. If the file has them in the expected order, this is
// a single linear scan.

/// Context for reading tensors sequentially from a .znap buffer.
fn SeqLoader(comptime ReaderType: type) type {
    return struct {
        reader: ReaderType,
        allocator: std.mem.Allocator,
        remaining: u32,

        const Self = @This();

        fn init(reader: ReaderType, allocator: std.mem.Allocator) !Self {
            const header = try readHeader(reader);
            return .{
                .reader = reader,
                .allocator = allocator,
                .remaining = header.num_tensors,
            };
        }

        fn readTensor(self: *Self, comptime Rows: usize, comptime Cols: usize, name: []const u8) !tensor.Tensor(Rows, Cols) {
            return readNamedTensor(Rows, Cols, self.reader, self.allocator, name, &self.remaining);
        }

        fn readVec(self: *Self, comptime N: usize, name: []const u8) !tensor.Vec(N) {
            return readNamedVec(N, self.reader, self.allocator, name, &self.remaining);
        }

        fn readArray(self: *Self, comptime N: usize, name: []const u8) ![N]f32 {
            const v = try self.readVec(N, name);
            return v.data[0];
        }
    };
}

// ── Policy loaders ───────────────────────────────────────────────

/// Load a LinearPolicy from a .znap file.
/// Expected tensors: "weights" (ActionDim x FeatureDim), "bias" (1 x ActionDim).
pub fn loadLinearPolicy(
    comptime FeatureDim: usize,
    comptime ActionDim: usize,
    path: []const u8,
) !policy.LinearPolicy(FeatureDim, ActionDim) {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return loadLinearPolicyFromReader(FeatureDim, ActionDim, file.reader());
}

/// Load a LinearPolicy from any reader (file, buffer, etc).
pub fn loadLinearPolicyFromReader(
    comptime FeatureDim: usize,
    comptime ActionDim: usize,
    reader: anytype,
) !policy.LinearPolicy(FeatureDim, ActionDim) {
    var buf: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    var loader = try SeqLoader(@TypeOf(reader)).init(reader, fba.allocator());
    const weights = try loader.readTensor(ActionDim, FeatureDim, "weights");
    const bias = try loader.readVec(ActionDim, "bias");
    return .{ .weights = weights, .bias = bias };
}

/// Load a DQNPolicy from a .znap file.
/// Expected tensors: "w1", "b1", "w2", "b2".
pub fn loadDQNPolicy(
    comptime FeatureDim: usize,
    comptime HiddenDim: usize,
    comptime ActionDim: usize,
    path: []const u8,
) !policy.DQNPolicy(FeatureDim, HiddenDim, ActionDim) {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return loadDQNPolicyFromReader(FeatureDim, HiddenDim, ActionDim, file.reader());
}

/// Load a DQNPolicy from any reader.
pub fn loadDQNPolicyFromReader(
    comptime FeatureDim: usize,
    comptime HiddenDim: usize,
    comptime ActionDim: usize,
    reader: anytype,
) !policy.DQNPolicy(FeatureDim, HiddenDim, ActionDim) {
    var buf: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    var loader = try SeqLoader(@TypeOf(reader)).init(reader, fba.allocator());
    const w1 = try loader.readTensor(HiddenDim, FeatureDim, "w1");
    const b1 = try loader.readVec(HiddenDim, "b1");
    const w2 = try loader.readTensor(ActionDim, HiddenDim, "w2");
    const b2 = try loader.readVec(ActionDim, "b2");
    return .{ .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2 };
}

/// Load a Transformer from a .znap file.
/// Expected tensors (in order):
///   "token_embedding"  (VocabSize x ModelDim)
///   For each layer l = 0..NumLayers-1:
///     "blocks.{l}.attn.wq"     (ModelDim x ModelDim)
///     "blocks.{l}.attn.wk"     (ModelDim x ModelDim)
///     "blocks.{l}.attn.wv"     (ModelDim x ModelDim)
///     "blocks.{l}.attn.wo"     (ModelDim x ModelDim)
///     "blocks.{l}.mlp.w1"      (HiddenDim x ModelDim)
///     "blocks.{l}.mlp.w_gate"  (HiddenDim x ModelDim)
///     "blocks.{l}.mlp.w2"      (ModelDim x HiddenDim)
///     "blocks.{l}.ln1_weight"  (1 x ModelDim)
///     "blocks.{l}.ln2_weight"  (1 x ModelDim)
///   "final_norm_weight"  (1 x ModelDim)
///   "output_proj"        (VocabSize x ModelDim)
pub fn loadTransformer(
    comptime ModelDim: usize,
    comptime NumHeads: usize,
    comptime NumLayers: usize,
    comptime VocabSize: usize,
    comptime MaxSeqLen: usize,
    path: []const u8,
) !transformer_mod.Transformer(ModelDim, NumHeads, NumLayers, VocabSize, MaxSeqLen) {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return loadTransformerFromReader(ModelDim, NumHeads, NumLayers, VocabSize, MaxSeqLen, file.reader());
}

/// Load a Transformer from any reader.
pub fn loadTransformerFromReader(
    comptime ModelDim: usize,
    comptime NumHeads: usize,
    comptime NumLayers: usize,
    comptime VocabSize: usize,
    comptime MaxSeqLen: usize,
    reader: anytype,
) !transformer_mod.Transformer(ModelDim, NumHeads, NumLayers, VocabSize, MaxSeqLen) {
    const Model = transformer_mod.Transformer(ModelDim, NumHeads, NumLayers, VocabSize, MaxSeqLen);
    const Block = transformer_mod.TransformerBlock(ModelDim, NumHeads, MaxSeqLen);
    const MLP = transformer_mod.SwiGLU(ModelDim);
    const HiddenDim = MLP.hidden_dim;

    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    var loader = try SeqLoader(@TypeOf(reader)).init(reader, fba.allocator());

    var model: Model = undefined;
    model.token_embedding = try loader.readTensor(VocabSize, ModelDim, "token_embedding");

    inline for (0..NumLayers) |l| {
        const prefix = comptime std.fmt.comptimePrint("blocks.{d}", .{l});
        model.blocks[l] = Block{
            .attn = .{
                .wq = try loader.readTensor(ModelDim, ModelDim, prefix ++ ".attn.wq"),
                .wk = try loader.readTensor(ModelDim, ModelDim, prefix ++ ".attn.wk"),
                .wv = try loader.readTensor(ModelDim, ModelDim, prefix ++ ".attn.wv"),
                .wo = try loader.readTensor(ModelDim, ModelDim, prefix ++ ".attn.wo"),
            },
            .mlp = .{
                .w1 = try loader.readTensor(HiddenDim, ModelDim, prefix ++ ".mlp.w1"),
                .w_gate = try loader.readTensor(HiddenDim, ModelDim, prefix ++ ".mlp.w_gate"),
                .w2 = try loader.readTensor(ModelDim, HiddenDim, prefix ++ ".mlp.w2"),
            },
            .ln1_weight = try loader.readArray(ModelDim, prefix ++ ".ln1_weight"),
            .ln2_weight = try loader.readArray(ModelDim, prefix ++ ".ln2_weight"),
        };
    }

    model.final_norm_weight = try loader.readArray(ModelDim, "final_norm_weight");
    model.output_proj = try loader.readTensor(VocabSize, ModelDim, "output_proj");

    return model;
}

// ── Writing ──────────────────────────────────────────────────────

/// Write the .znap file header.
///
/// Python equivalent for writing .znap files:
///
///   ```python
///   import struct
///   import numpy as np
///   import torch
///
///   def save_znap(path: str, tensors: dict[str, np.ndarray]):
///       """Export named tensors to .znap format.
///
///       Args:
///           path: Output file path (e.g. "model.znap")
///           tensors: Dict mapping tensor names to numpy arrays.
///                    Arrays must be float32.
///
///       Example — export a PyTorch LinearPolicy:
///           save_znap("policy.znap", {
///               "weights": model.linear.weight.detach().numpy(),  # (A, F)
///               "bias": model.linear.bias.detach().numpy().reshape(1, -1),  # (1, A)
///           })
///
///       Example — export a PyTorch DQN:
///           save_znap("dqn.znap", {
///               "w1": model.fc1.weight.detach().numpy(),
///               "b1": model.fc1.bias.detach().numpy().reshape(1, -1),
///               "w2": model.fc2.weight.detach().numpy(),
///               "b2": model.fc2.bias.detach().numpy().reshape(1, -1),
///           })
///       """
///       with open(path, 'wb') as f:
///           # Header
///           f.write(b'ZNAP')
///           f.write(struct.pack('<I', 1))          # version
///           f.write(struct.pack('<I', len(tensors)))  # num_tensors
///           f.write(struct.pack('<I', 0))          # metadata_len
///
///           for name, arr in tensors.items():
///               arr = arr.astype(np.float32)
///               name_bytes = name.encode('utf-8')
///               f.write(struct.pack('<H', len(name_bytes)))
///               f.write(name_bytes)
///               f.write(struct.pack('B', arr.ndim))
///               for d in arr.shape:
///                   f.write(struct.pack('<I', d))
///               f.write(struct.pack('B', 0))  # dtype = f32
///               f.write(arr.tobytes())
///   ```
pub fn writeHeader(writer: anytype, num_tensors: u32) !void {
    try writer.writeAll(&MAGIC);
    try writer.writeInt(u32, FORMAT_VERSION, .little);
    try writer.writeInt(u32, num_tensors, .little);
    try writer.writeInt(u32, 0, .little); // metadata_len reserved
}

/// Write a single named tensor entry (header + data).
pub fn writeTensor(
    writer: anytype,
    name: []const u8,
    shape: []const u32,
    data: []const f32,
) !void {
    // Name
    try writer.writeInt(u16, @intCast(name.len), .little);
    try writer.writeAll(name);

    // Shape
    try writer.writeByte(@intCast(shape.len));
    for (shape) |d| {
        try writer.writeInt(u32, d, .little);
    }

    // Dtype (always f32 for now)
    try writer.writeByte(@intFromEnum(DType.f32));

    // Data as little-endian bytes
    const byte_slice = std.mem.sliceAsBytes(data);
    try writer.writeAll(byte_slice);
}

/// Write a Tensor(R, C) as a named entry.
pub fn writeTensorMatrix(
    comptime Rows: usize,
    comptime Cols: usize,
    writer: anytype,
    name: []const u8,
    t: *const tensor.Tensor(Rows, Cols),
) !void {
    // Flatten to contiguous slice
    const flat: *const [Rows * Cols]f32 = @ptrCast(&t.data);
    const shape = [_]u32{ @intCast(Rows), @intCast(Cols) };
    try writeTensor(writer, name, &shape, flat);
}

/// Write a Vec(N) = Tensor(1, N) as a named entry.
pub fn writeTensorVec(
    comptime N: usize,
    writer: anytype,
    name: []const u8,
    v: *const tensor.Vec(N),
) !void {
    writeTensorMatrix(1, N, writer, name, v) catch |e| return e;
}

/// Write a raw [N]f32 array as a named 1D tensor entry.
pub fn writeArray(
    comptime N: usize,
    writer: anytype,
    name: []const u8,
    arr: *const [N]f32,
) !void {
    const shape = [_]u32{ 1, @intCast(N) };
    try writeTensor(writer, name, &shape, arr);
}

// ── Policy savers ────────────────────────────────────────────────

/// Save a LinearPolicy to a .znap file.
pub fn saveLinearPolicy(
    comptime FeatureDim: usize,
    comptime ActionDim: usize,
    p: *const policy.LinearPolicy(FeatureDim, ActionDim),
    path: []const u8,
) !void {
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try saveLinearPolicyToWriter(FeatureDim, ActionDim, p, file.writer());
}

/// Save a LinearPolicy to any writer.
pub fn saveLinearPolicyToWriter(
    comptime FeatureDim: usize,
    comptime ActionDim: usize,
    p: *const policy.LinearPolicy(FeatureDim, ActionDim),
    writer: anytype,
) !void {
    try writeHeader(writer, 2);
    try writeTensorMatrix(ActionDim, FeatureDim, writer, "weights", &p.weights);
    try writeTensorMatrix(1, ActionDim, writer, "bias", &p.bias);
}

/// Save a DQNPolicy to a .znap file.
pub fn saveDQNPolicy(
    comptime FeatureDim: usize,
    comptime HiddenDim: usize,
    comptime ActionDim: usize,
    p: *const policy.DQNPolicy(FeatureDim, HiddenDim, ActionDim),
    path: []const u8,
) !void {
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try saveDQNPolicyToWriter(FeatureDim, HiddenDim, ActionDim, p, file.writer());
}

/// Save a DQNPolicy to any writer.
pub fn saveDQNPolicyToWriter(
    comptime FeatureDim: usize,
    comptime HiddenDim: usize,
    comptime ActionDim: usize,
    p: *const policy.DQNPolicy(FeatureDim, HiddenDim, ActionDim),
    writer: anytype,
) !void {
    try writeHeader(writer, 4);
    try writeTensorMatrix(HiddenDim, FeatureDim, writer, "w1", &p.w1);
    try writeTensorMatrix(1, HiddenDim, writer, "b1", &p.b1);
    try writeTensorMatrix(ActionDim, HiddenDim, writer, "w2", &p.w2);
    try writeTensorMatrix(1, ActionDim, writer, "b2", &p.b2);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

/// Helper: write a .znap buffer in memory using an ArrayList.
fn testWriteToBuffer(comptime writeFn: anytype, args: anytype) ![]u8 {
    var list = std.ArrayList(u8).init(std.testing.allocator);
    errdefer list.deinit();
    const writer = list.writer();
    try @call(.auto, writeFn, .{writer} ++ args);
    return list.toOwnedSlice();
}

test "header round-trip" {
    // Write a header
    var buf: [16]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeHeader(fbs.writer(), 3);

    // Read it back
    fbs.pos = 0;
    const header = try readHeader(fbs.reader());
    try std.testing.expectEqual(@as(u32, 1), header.version);
    try std.testing.expectEqual(@as(u32, 3), header.num_tensors);
    try std.testing.expectEqual(@as(u32, 0), header.metadata_len);
}

test "invalid magic" {
    var buf = [_]u8{ 'B', 'A', 'D', '!', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var fbs = std.io.fixedBufferStream(&buf);
    try std.testing.expectError(WeightError.InvalidMagic, readHeader(fbs.reader()));
}

test "loadTensorFromSlice" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const t = try loadTensorFromSlice(2, 3, &data);
    try std.testing.expectEqual(@as(f32, 1.0), t.data[0][0]);
    try std.testing.expectEqual(@as(f32, 4.0), t.data[1][0]);
    try std.testing.expectEqual(@as(f32, 6.0), t.data[1][2]);
}

test "loadTensorFromSlice shape mismatch" {
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(WeightError.ShapeMismatch, loadTensorFromSlice(2, 3, &data));
}

test "linear policy round-trip via buffer" {
    const F = 4;
    const A = 3;
    const P = policy.LinearPolicy(F, A);

    // Create a policy with known weights
    var p = P.init();
    p.weights.data[0] = .{ 1.0, 2.0, 3.0, 4.0 };
    p.weights.data[1] = .{ 5.0, 6.0, 7.0, 8.0 };
    p.weights.data[2] = .{ 9.0, 10.0, 11.0, 12.0 };
    p.bias.data[0] = .{ 0.1, 0.2, 0.3 };

    // Write to buffer
    var list = std.ArrayList(u8).init(std.testing.allocator);
    defer list.deinit();
    try saveLinearPolicyToWriter(F, A, &p, list.writer());

    // Read back
    var fbs = std.io.fixedBufferStream(list.items);
    const loaded = try loadLinearPolicyFromReader(F, A, fbs.reader());

    // Verify weights match
    for (0..A) |r| {
        for (0..F) |c| {
            try std.testing.expectEqual(p.weights.data[r][c], loaded.weights.data[r][c]);
        }
    }
    for (0..A) |i| {
        try std.testing.expectEqual(p.bias.data[0][i], loaded.bias.data[0][i]);
    }
}

test "DQN policy round-trip via buffer" {
    const F = 4;
    const H = 8;
    const A = 3;
    const D = policy.DQNPolicy(F, H, A);

    // Create with Xavier init for non-trivial weights
    const d = D.initXavier(42);

    // Write to buffer
    var list = std.ArrayList(u8).init(std.testing.allocator);
    defer list.deinit();
    try saveDQNPolicyToWriter(F, H, A, &d, list.writer());

    // Read back
    var fbs = std.io.fixedBufferStream(list.items);
    const loaded = try loadDQNPolicyFromReader(F, H, A, fbs.reader());

    // Verify all weights match
    for (0..H) |r| {
        for (0..F) |c| {
            try std.testing.expectEqual(d.w1.data[r][c], loaded.w1.data[r][c]);
        }
    }
    for (0..H) |i| {
        try std.testing.expectEqual(d.b1.data[0][i], loaded.b1.data[0][i]);
    }
    for (0..A) |r| {
        for (0..H) |c| {
            try std.testing.expectEqual(d.w2.data[r][c], loaded.w2.data[r][c]);
        }
    }
    for (0..A) |i| {
        try std.testing.expectEqual(d.b2.data[0][i], loaded.b2.data[0][i]);
    }
}

test "loaded policy produces same inference" {
    const F = 4;
    const A = 3;
    const P = policy.LinearPolicy(F, A);

    // Train a policy so it has non-trivial weights
    var p = P.init();
    var features = [_]f32{ 0.5, 0.3, 0.8, 0.1 };
    for (0..50) |_| p.update(&features, 1, 1.0, 0.05);

    // Serialize
    var list = std.ArrayList(u8).init(std.testing.allocator);
    defer list.deinit();
    try saveLinearPolicyToWriter(F, A, &p, list.writer());

    // Deserialize
    var fbs = std.io.fixedBufferStream(list.items);
    const loaded = try loadLinearPolicyFromReader(F, A, fbs.reader());

    // Both should produce identical results
    const r1 = p.evaluate(&features);
    const r2 = loaded.evaluate(&features);
    try std.testing.expectEqual(r1.action, r2.action);
    try std.testing.expectApproxEqAbs(r1.confidence, r2.confidence, 1e-6);
    for (0..A) |i| {
        try std.testing.expectApproxEqAbs(r1.scores[i], r2.scores[i], 1e-6);
    }
}

test "write and read single tensor entry" {
    // Write a 2x3 tensor
    var list = std.ArrayList(u8).init(std.testing.allocator);
    defer list.deinit();
    const writer = list.writer();

    try writeHeader(writer, 1);
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const shape = [_]u32{ 2, 3 };
    try writeTensor(writer, "test_tensor", &shape, &data);

    // Read back
    var fbs = std.io.fixedBufferStream(list.items);
    const reader = fbs.reader();

    const header = try readHeader(reader);
    try std.testing.expectEqual(@as(u32, 1), header.num_tensors);

    const entry = try readTensorMeta(reader, std.testing.allocator);
    defer std.testing.allocator.free(entry.name);

    try std.testing.expectEqualStrings("test_tensor", entry.name);
    try std.testing.expectEqual(@as(u8, 2), entry.ndim);
    try std.testing.expectEqual(@as(u32, 2), entry.dims[0]);
    try std.testing.expectEqual(@as(u32, 3), entry.dims[1]);
    try std.testing.expectEqual(DType.f32, entry.dtype);

    var read_data: [6]f32 = undefined;
    try readF32Data(reader, &read_data);
    for (0..6) |i| {
        try std.testing.expectEqual(data[i], read_data[i]);
    }
}

test "multiple tensors sequential read" {
    var list = std.ArrayList(u8).init(std.testing.allocator);
    defer list.deinit();
    const writer = list.writer();

    try writeHeader(writer, 2);
    try writeTensor(writer, "first", &.{ 1, 2 }, &.{ 10.0, 20.0 });
    try writeTensor(writer, "second", &.{ 1, 3 }, &.{ 30.0, 40.0, 50.0 });

    var fbs = std.io.fixedBufferStream(list.items);
    const reader = fbs.reader();
    var remaining: u32 = undefined;

    const header = try readHeader(reader);
    remaining = header.num_tensors;

    const t1 = try readNamedTensor(1, 2, reader, std.testing.allocator, "first", &remaining);
    try std.testing.expectEqual(@as(f32, 10.0), t1.data[0][0]);
    try std.testing.expectEqual(@as(f32, 20.0), t1.data[0][1]);

    const t2 = try readNamedTensor(1, 3, reader, std.testing.allocator, "second", &remaining);
    try std.testing.expectEqual(@as(f32, 30.0), t2.data[0][0]);
    try std.testing.expectEqual(@as(f32, 50.0), t2.data[0][2]);
}
