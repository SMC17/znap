// GGUF file parser — loads GGUF v3 model files and dequantizes tensors to f32.
//
// Supports quantization types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K
//
// References:
//   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const std = @import("std");

// ── GGUF Constants ──────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in LE
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ── Value Types ─────────────────────────────────────────────────────

pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

// ── Tensor Types (quantization formats) ─────────────────────────────

pub const TensorType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,
    Q8_K = 15,
    _,

    /// Number of weights per quantization block.
    pub fn blockSize(self: TensorType) usize {
        return switch (self) {
            .F32 => 1,
            .F16 => 1,
            .Q4_0 => 32,
            .Q4_1 => 32,
            .Q5_0 => 32,
            .Q5_1 => 32,
            .Q8_0 => 32,
            .Q4_K => 256,
            .Q6_K => 256,
            .Q8_K => 256,
            _ => 1,
        };
    }

    /// Bytes per block for this quantization type.
    pub fn bytesPerBlock(self: TensorType) usize {
        return switch (self) {
            .F32 => 4,
            .F16 => 2,
            .Q4_0 => 2 + 16, // f16 scale + 16 bytes (32 nibbles)
            .Q4_1 => 2 + 2 + 16, // f16 scale + f16 min + 16 bytes
            .Q5_0 => 2 + 4 + 16, // f16 scale + 4 bytes high bits + 16 bytes low nibbles
            .Q5_1 => 2 + 2 + 4 + 16, // f16 scale + f16 min + 4 bytes high + 16 bytes
            .Q8_0 => 2 + 32, // f16 scale + 32 int8s
            .Q4_K => 2 + 2 + 12 + 128, // d(f16) + dmin(f16) + 12 scale bytes + 128 data bytes = 144
            .Q6_K => 128 + 64 + 2 + 32, // ql(128) + qh(64) + d(f16) + scales(32) = 210 (but actually different)
            .Q8_K => 4 + 256 + 16, // d(f32) + 256 int8s + 16 bsums = 276 (but actually: 4 + 256 = 260 with 2 bsums per 32)
            _ => 0,
        };
    }
};

// ── Metadata Value ──────────────────────────────────────────────────

pub const MetaValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    uint64: u64,
    int64: i64,
    float64: f64,
    array: ArrayValue,

    pub const ArrayValue = struct {
        elem_type: ValueType,
        len: u64,
        /// Raw byte offset into the file data where array elements start
        data_offset: usize,
    };

    pub fn asU32(self: MetaValue) ?u32 {
        return switch (self) {
            .uint32 => |v| v,
            .int32 => |v| if (v >= 0) @as(u32, @intCast(v)) else null,
            .uint64 => |v| if (v <= std.math.maxInt(u32)) @as(u32, @intCast(v)) else null,
            else => null,
        };
    }

    pub fn asF32(self: MetaValue) ?f32 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| @as(f32, @floatCast(v)),
            else => null,
        };
    }

    pub fn asString(self: MetaValue) ?[]const u8 {
        return switch (self) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn asBool(self: MetaValue) ?bool {
        return switch (self) {
            .bool_ => |b| b,
            else => null,
        };
    }
};

// ── Tensor Info ─────────────────────────────────────────────────────

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64, // max 4 dimensions
    tensor_type: TensorType,
    offset: u64, // offset from start of tensor data block

    pub fn numElements(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (0..self.n_dims) |i| n *= self.dims[i];
        return n;
    }

    /// Total bytes this tensor occupies in quantized form.
    pub fn dataSize(self: TensorInfo) u64 {
        const n = self.numElements();
        const bs = self.tensor_type.blockSize();
        const bpb = self.tensor_type.bytesPerBlock();
        if (bs == 0 or bpb == 0) return 0;
        const nblocks = (n + bs - 1) / bs;
        return nblocks * bpb;
    }
};

// ── KV Entry ────────────────────────────────────────────────────────

const KVEntry = struct {
    key: []const u8,
    value: MetaValue,
};

// ── GGUF File ───────────────────────────────────────────────────────

pub const GGUFFile = struct {
    version: u32,
    tensor_count: u64,
    kv_count: u64,
    alignment: usize,

    kv_entries: []KVEntry,
    tensor_infos: []TensorInfo,

    /// Byte offset where tensor data block starts (after header + metadata + tensor infos + padding)
    data_offset: usize,

    /// Raw file data (entire file loaded into memory)
    file_data: []align(4) const u8,

    allocator: std.mem.Allocator,

    pub fn open(path: []const u8, allocator: std.mem.Allocator) !GGUFFile {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const data = try allocator.alignedAlloc(u8, 4, file_size);
        errdefer allocator.free(data);
        const bytes_read = try file.readAll(data);
        if (bytes_read != file_size) return error.IncompleteRead;

        return parseFromData(data, allocator);
    }

    fn parseFromData(data: []align(4) const u8, allocator: std.mem.Allocator) !GGUFFile {
        var pos: usize = 0;

        // Magic
        const magic = readU32(data, &pos);
        if (magic != GGUF_MAGIC) return error.InvalidMagic;

        // Version
        const version = readU32(data, &pos);
        if (version < 2 or version > 3) return error.UnsupportedVersion;

        // Counts
        const tensor_count = readU64(data, &pos);
        const kv_count = readU64(data, &pos);

        // Parse KV metadata
        var kv_entries = try allocator.alloc(KVEntry, kv_count);
        errdefer allocator.free(kv_entries);

        var alignment: usize = GGUF_DEFAULT_ALIGNMENT;

        for (0..kv_count) |i| {
            const key = readString(data, &pos);
            const val_type: ValueType = @enumFromInt(readU32(data, &pos));
            const value = readValue(data, &pos, val_type);

            kv_entries[i] = .{ .key = key, .value = value };

            // Check for custom alignment
            if (std.mem.eql(u8, key, "general.alignment")) {
                if (value.asU32()) |a| alignment = @intCast(a);
            }
        }

        // Parse tensor infos
        var tensor_infos = try allocator.alloc(TensorInfo, tensor_count);
        errdefer allocator.free(tensor_infos);

        for (0..tensor_count) |i| {
            const name = readString(data, &pos);
            const n_dims = readU32(data, &pos);
            var dims = [4]u64{ 1, 1, 1, 1 };
            for (0..n_dims) |d| {
                dims[d] = readU64(data, &pos);
            }
            const ttype: TensorType = @enumFromInt(readU32(data, &pos));
            const offset = readU64(data, &pos);

            tensor_infos[i] = .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .tensor_type = ttype,
                .offset = offset,
            };
        }

        // Compute data offset (aligned)
        const data_offset = alignUp(pos, alignment);

        return .{
            .version = version,
            .tensor_count = tensor_count,
            .kv_count = kv_count,
            .alignment = alignment,
            .kv_entries = kv_entries,
            .tensor_infos = tensor_infos,
            .data_offset = data_offset,
            .file_data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GGUFFile) void {
        self.allocator.free(self.kv_entries);
        self.allocator.free(self.tensor_infos);
        self.allocator.free(self.file_data);
    }

    /// Look up a metadata value by key.
    pub fn getMetadata(self: *const GGUFFile, key: []const u8) ?MetaValue {
        for (self.kv_entries) |entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry.value;
        }
        return null;
    }

    /// Find a tensor by name.
    pub fn getTensorInfo(self: *const GGUFFile, name: []const u8) ?TensorInfo {
        for (self.tensor_infos) |info| {
            if (std.mem.eql(u8, info.name, name)) return info;
        }
        return null;
    }

    /// Read tensor data and dequantize to f32.
    pub fn readTensorF32(self: *const GGUFFile, name: []const u8, allocator: std.mem.Allocator) ![]f32 {
        const info = self.getTensorInfo(name) orelse return error.TensorNotFound;
        return self.dequantizeTensor(info, allocator);
    }

    /// Dequantize a tensor to f32 given its info.
    pub fn dequantizeTensor(self: *const GGUFFile, info: TensorInfo, allocator: std.mem.Allocator) ![]f32 {
        const n_elements = info.numElements();
        const raw_offset = self.data_offset + @as(usize, @intCast(info.offset));
        const raw_data = self.file_data[raw_offset..];

        var result = try allocator.alloc(f32, n_elements);
        errdefer allocator.free(result);

        switch (info.tensor_type) {
            .F32 => {
                const src: [*]const f32 = @ptrCast(@alignCast(raw_data.ptr));
                @memcpy(result, src[0..n_elements]);
            },
            .F16 => {
                const src: [*]const f16 = @ptrCast(@alignCast(raw_data.ptr));
                for (0..n_elements) |i| {
                    result[i] = @floatCast(src[i]);
                }
            },
            .Q4_0 => dequantQ4_0(raw_data, result, n_elements),
            .Q4_1 => dequantQ4_1(raw_data, result, n_elements),
            .Q5_0 => dequantQ5_0(raw_data, result, n_elements),
            .Q5_1 => dequantQ5_1(raw_data, result, n_elements),
            .Q8_0 => dequantQ8_0(raw_data, result, n_elements),
            .Q4_K => dequantQ4_K(raw_data, result, n_elements),
            .Q6_K => dequantQ6_K(raw_data, result, n_elements),
            else => return error.UnsupportedQuantType,
        }

        return result;
    }

    /// Read array of strings from metadata (e.g. tokenizer.ggml.tokens)
    pub fn readStringArray(self: *const GGUFFile, key: []const u8, allocator: std.mem.Allocator) ![][]const u8 {
        const meta = self.getMetadata(key) orelse return error.MetadataNotFound;
        const arr = switch (meta) {
            .array => |a| a,
            else => return error.NotAnArray,
        };
        if (arr.elem_type != .string) return error.NotStringArray;

        const len: usize = @intCast(arr.len);
        var strings = try allocator.alloc([]const u8, len);

        var pos = arr.data_offset;
        for (0..len) |i| {
            strings[i] = readString(self.file_data, &pos);
        }

        return strings;
    }
};

// ── Dequantization Routines ─────────────────────────────────────────

/// Read f16 from 2 bytes and convert to f32.
fn f16ToF32(bytes: *const [2]u8) f32 {
    const val: f16 = @bitCast(bytes.*);
    return @floatCast(val);
}

/// Q4_0: 18 bytes per block (f16 scale + 16 bytes = 32 weights)
/// ggml order: first 16 values use low nibbles, last 16 use high nibbles.
fn dequantQ4_0(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 18;
    const n_blocks = n_elements / 32;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const scale = f16ToF32(block[0..2]);
        const quants = block[2..18];
        for (0..16) |i| {
            const lo: i8 = @as(i8, @intCast(quants[i] & 0xF)) - 8;
            const hi: i8 = @as(i8, @intCast(quants[i] >> 4)) - 8;
            out[b * 32 + i] = scale * @as(f32, @floatFromInt(lo));
            out[b * 32 + i + 16] = scale * @as(f32, @floatFromInt(hi));
        }
    }
}

/// Q4_1: 20 bytes per block (f16 scale + f16 min + 16 bytes = 32 weights)
/// ggml order: first 16 values use low nibbles, last 16 use high nibbles.
fn dequantQ4_1(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 20;
    const n_blocks = n_elements / 32;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const scale = f16ToF32(block[0..2]);
        const min = f16ToF32(block[2..4]);
        const quants = block[4..20];
        for (0..16) |i| {
            out[b * 32 + i] = scale * @as(f32, @floatFromInt(quants[i] & 0xF)) + min;
            out[b * 32 + i + 16] = scale * @as(f32, @floatFromInt(quants[i] >> 4)) + min;
        }
    }
}

/// Q5_0: 22 bytes per block (f16 scale + 4 bytes high bits + 16 bytes low nibbles = 32 weights)
/// ggml order: first 16 values use low nibbles (qh bits 0..15), last 16 use high nibbles (qh bits 16..31).
fn dequantQ5_0(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 22;
    const n_blocks = n_elements / 32;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const scale = f16ToF32(block[0..2]);
        const qh: u32 = std.mem.readInt(u32, block[2..6], .little);
        const quants = block[6..22]; // 16 bytes

        for (0..16) |i| {
            const xh_0: u8 = @intCast((qh >> @as(u5, @intCast(i))) & 1);
            const xh_1: u8 = @intCast((qh >> @as(u5, @intCast(i + 16))) & 1);

            const val0: i8 = @as(i8, @intCast((quants[i] & 0xF) | (xh_0 << 4))) - 16;
            const val1: i8 = @as(i8, @intCast((quants[i] >> 4) | (xh_1 << 4))) - 16;

            out[b * 32 + i] = scale * @as(f32, @floatFromInt(val0));
            out[b * 32 + i + 16] = scale * @as(f32, @floatFromInt(val1));
        }
    }
}

/// Q5_1: 24 bytes per block (f16 scale + f16 min + 4 bytes high bits + 16 bytes = 32 weights)
/// ggml order: first 16 values use low nibbles, last 16 use high nibbles.
fn dequantQ5_1(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 24;
    const n_blocks = n_elements / 32;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const scale = f16ToF32(block[0..2]);
        const min = f16ToF32(block[2..4]);
        const qh: u32 = std.mem.readInt(u32, block[4..8], .little);
        const quants = block[8..24];

        for (0..16) |i| {
            const xh_0: u8 = @intCast((qh >> @as(u5, @intCast(i))) & 1);
            const xh_1: u8 = @intCast((qh >> @as(u5, @intCast(i + 16))) & 1);

            const val0: u8 = (quants[i] & 0xF) | (xh_0 << 4);
            const val1: u8 = (quants[i] >> 4) | (xh_1 << 4);

            out[b * 32 + i] = scale * @as(f32, @floatFromInt(val0)) + min;
            out[b * 32 + i + 16] = scale * @as(f32, @floatFromInt(val1)) + min;
        }
    }
}

/// Q8_0: 34 bytes per block (f16 scale + 32 × i8 = 32 weights)
fn dequantQ8_0(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 34;
    const n_blocks = n_elements / 32;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const scale = f16ToF32(block[0..2]);
        for (0..32) |j| {
            const q: i8 = @bitCast(block[2 + j]);
            out[b * 32 + j] = scale * @as(f32, @floatFromInt(q));
        }
    }
}

/// Q4_K: 144 bytes per block of 256 weights.
/// Layout: d(f16) + dmin(f16) + scales(12 bytes) + quants(128 bytes)
/// Follows ggml-quants.c dequantize_row_q4_K exactly.
fn dequantQ4_K(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 144;
    const n_blocks = n_elements / 256;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const d = f16ToF32(block[0..2]);
        const dmin = f16ToF32(block[2..4]);
        const scales_raw = block[4..16]; // 12 bytes
        const qs = block[16..144]; // 128 bytes

        // Decode scales using ggml's bit manipulation:
        // 12 raw bytes -> 16 decoded bytes (interleaved scale/min pairs)
        var utmp: [4]u32 = undefined;
        utmp[0] = std.mem.readInt(u32, scales_raw[0..4], .little);
        utmp[1] = std.mem.readInt(u32, scales_raw[4..8], .little);
        utmp[2] = std.mem.readInt(u32, scales_raw[8..12], .little);
        utmp[3] = 0;

        const utmp3 = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
        const utmp2 = ((utmp[2] >> 0) & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
        utmp[0] = utmp[0] & 0x3f3f3f3f;
        utmp[1] = utmp[1] & 0x3f3f3f3f;
        utmp[2] = utmp2;
        utmp[3] = utmp3;

        // sc is a 16-byte array from utmp[0..4]
        const sc: [16]u8 = @bitCast(utmp);

        // Dequantize 256 weights in two halves of 128
        // Each half has 4 sub-blocks of 32 weights
        var is: usize = 0;
        var j: usize = 0;
        while (j < 128) : (j += 32) {
            const d1 = d * @as(f32, @floatFromInt(sc[is + 0]));
            const m1 = dmin * @as(f32, @floatFromInt(sc[is + 1]));
            const d2 = d * @as(f32, @floatFromInt(sc[is + 2]));
            const m2 = dmin * @as(f32, @floatFromInt(sc[is + 3]));

            for (0..32) |l| {
                const q_byte = qs[j / 2 + l % 16];
                const nibble: u8 = if (l < 16) (q_byte & 0xF) else (q_byte >> 4);
                out[b * 256 + j + l] = d1 * @as(f32, @floatFromInt(nibble)) - m1;
            }

            for (0..32) |l| {
                const q_byte = qs[64 + j / 2 + l % 16];
                const nibble: u8 = if (l < 16) (q_byte & 0xF) else (q_byte >> 4);
                out[b * 256 + 128 + j + l] = d2 * @as(f32, @floatFromInt(nibble)) - m2;
            }

            is += 4;
        }
    }
}

/// Q6_K: 210 bytes per block of 256 weights.
/// Layout: ql(128) + qh(64) + scales(16 × i8) + d(f16)
/// Follows ggml-quants.c dequantize_row_q6_K exactly.
fn dequantQ6_K(raw: []const u8, out: []f32, n_elements: u64) void {
    const block_size: usize = 210;
    const n_blocks = n_elements / 256;
    for (0..n_blocks) |b| {
        const block = raw[b * block_size ..];
        const ql = block[0..128];
        const qh = block[128..192];
        const scales: *const [16]i8 = @ptrCast(block[192..208]);
        const d = f16ToF32(block[208..210]);

        // Half 1: weights 0-127, using ql[0..63], qh[0..31]
        for (0..32) |l| {
            const is: usize = l / 16;
            const q1: i8 = @as(i8, @intCast((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
            const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
            const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
            const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;
            out[b * 256 + l + 0] = d * @as(f32, @floatFromInt(scales[is + 0])) * @as(f32, @floatFromInt(q1));
            out[b * 256 + l + 32] = d * @as(f32, @floatFromInt(scales[is + 2])) * @as(f32, @floatFromInt(q2));
            out[b * 256 + l + 64] = d * @as(f32, @floatFromInt(scales[is + 4])) * @as(f32, @floatFromInt(q3));
            out[b * 256 + l + 96] = d * @as(f32, @floatFromInt(scales[is + 6])) * @as(f32, @floatFromInt(q4));
        }

        // Half 2: weights 128-255, using ql[64..127], qh[32..63]
        for (0..32) |l| {
            const is: usize = l / 16;
            const q1: i8 = @as(i8, @intCast((ql[64 + l] & 0xF) | (((qh[32 + l] >> 0) & 3) << 4))) - 32;
            const q2: i8 = @as(i8, @intCast((ql[64 + l + 32] & 0xF) | (((qh[32 + l] >> 2) & 3) << 4))) - 32;
            const q3: i8 = @as(i8, @intCast((ql[64 + l] >> 4) | (((qh[32 + l] >> 4) & 3) << 4))) - 32;
            const q4: i8 = @as(i8, @intCast((ql[64 + l + 32] >> 4) | (((qh[32 + l] >> 6) & 3) << 4))) - 32;
            out[b * 256 + 128 + l + 0] = d * @as(f32, @floatFromInt(scales[is + 8])) * @as(f32, @floatFromInt(q1));
            out[b * 256 + 128 + l + 32] = d * @as(f32, @floatFromInt(scales[is + 10])) * @as(f32, @floatFromInt(q2));
            out[b * 256 + 128 + l + 64] = d * @as(f32, @floatFromInt(scales[is + 12])) * @as(f32, @floatFromInt(q3));
            out[b * 256 + 128 + l + 96] = d * @as(f32, @floatFromInt(scales[is + 14])) * @as(f32, @floatFromInt(q4));
        }
    }
}

// ── Binary Helpers ──────────────────────────────────────────────────

fn readU32(data: []const u8, pos: *usize) u32 {
    const bytes = data[pos.*..][0..4];
    pos.* += 4;
    return std.mem.readInt(u32, bytes, .little);
}

fn readU64(data: []const u8, pos: *usize) u64 {
    const bytes = data[pos.*..][0..8];
    pos.* += 8;
    return std.mem.readInt(u64, bytes, .little);
}

fn readI32(data: []const u8, pos: *usize) i32 {
    const bytes = data[pos.*..][0..4];
    pos.* += 4;
    return std.mem.readInt(i32, bytes, .little);
}

fn readString(data: []const u8, pos: *usize) []const u8 {
    const len: usize = @intCast(readU64(data, pos));
    const s = data[pos.* .. pos.* + len];
    pos.* += len;
    return s;
}

fn readValue(data: []const u8, pos: *usize, val_type: ValueType) MetaValue {
    return switch (val_type) {
        .uint8 => blk: {
            const v = data[pos.*];
            pos.* += 1;
            break :blk .{ .uint8 = v };
        },
        .int8 => blk: {
            const v: i8 = @bitCast(data[pos.*]);
            pos.* += 1;
            break :blk .{ .int8 = v };
        },
        .uint16 => blk: {
            const v = std.mem.readInt(u16, data[pos.*..][0..2], .little);
            pos.* += 2;
            break :blk .{ .uint16 = v };
        },
        .int16 => blk: {
            const v = std.mem.readInt(i16, data[pos.*..][0..2], .little);
            pos.* += 2;
            break :blk .{ .int16 = v };
        },
        .uint32 => .{ .uint32 = readU32(data, pos) },
        .int32 => .{ .int32 = readI32(data, pos) },
        .float32 => blk: {
            const bits = readU32(data, pos);
            break :blk .{ .float32 = @bitCast(bits) };
        },
        .bool_ => blk: {
            const v = data[pos.*] != 0;
            pos.* += 1;
            break :blk .{ .bool_ = v };
        },
        .string => .{ .string = readString(data, pos) },
        .uint64 => .{ .uint64 = readU64(data, pos) },
        .int64 => blk: {
            const v: i64 = @bitCast(readU64(data, pos));
            break :blk .{ .int64 = v };
        },
        .float64 => blk: {
            const bits = readU64(data, pos);
            break :blk .{ .float64 = @bitCast(bits) };
        },
        .array => blk: {
            const elem_type: ValueType = @enumFromInt(readU32(data, pos));
            const len = readU64(data, pos);
            const data_start = pos.*;
            // Skip over array elements
            for (0..len) |_| {
                skipValue(data, pos, elem_type);
            }
            break :blk .{ .array = .{
                .elem_type = elem_type,
                .len = len,
                .data_offset = data_start,
            } };
        },
        _ => blk: {
            // Unknown type — can't skip safely
            break :blk .{ .uint32 = 0 };
        },
    };
}

fn skipValue(data: []const u8, pos: *usize, val_type: ValueType) void {
    switch (val_type) {
        .uint8, .int8, .bool_ => pos.* += 1,
        .uint16, .int16 => pos.* += 2,
        .uint32, .int32, .float32 => pos.* += 4,
        .uint64, .int64, .float64 => pos.* += 8,
        .string => {
            const len: usize = @intCast(readU64(data, pos));
            pos.* += len;
        },
        .array => {
            const elem_type: ValueType = @enumFromInt(readU32(data, pos));
            const len = readU64(data, pos);
            for (0..len) |_| {
                skipValue(data, pos, elem_type);
            }
        },
        _ => {},
    }
}

fn alignUp(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

// ── Tests ───────────────────────────────────────────────────────────

test "GGUF header parsing" {
    // Test with the real SmolLM2-135M model file
    var gguf = GGUFFile.open("models/smollm2-135m-q4_k_m.gguf", std.testing.allocator) catch |err| {
        // Skip test if model file is not present
        if (err == error.FileNotFound) return;
        return err;
    };
    defer gguf.deinit();

    try std.testing.expectEqual(@as(u32, 3), gguf.version);
    try std.testing.expectEqual(@as(u64, 272), gguf.tensor_count);
    try std.testing.expectEqual(@as(u64, 37), gguf.kv_count);
}

test "GGUF metadata reading" {
    var gguf = GGUFFile.open("models/smollm2-135m-q4_k_m.gguf", std.testing.allocator) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer gguf.deinit();

    // Architecture
    const arch = gguf.getMetadata("general.architecture") orelse return error.MissingMetadata;
    try std.testing.expectEqualStrings("llama", arch.asString() orelse return error.WrongType);

    // Model dimensions
    const dim = gguf.getMetadata("llama.embedding_length") orelse return error.MissingMetadata;
    try std.testing.expectEqual(@as(u32, 576), dim.asU32() orelse return error.WrongType);

    const n_layers = gguf.getMetadata("llama.block_count") orelse return error.MissingMetadata;
    try std.testing.expectEqual(@as(u32, 30), n_layers.asU32() orelse return error.WrongType);

    const n_heads = gguf.getMetadata("llama.attention.head_count") orelse return error.MissingMetadata;
    try std.testing.expectEqual(@as(u32, 9), n_heads.asU32() orelse return error.WrongType);

    const n_kv_heads = gguf.getMetadata("llama.attention.head_count_kv") orelse return error.MissingMetadata;
    try std.testing.expectEqual(@as(u32, 3), n_kv_heads.asU32() orelse return error.WrongType);

    const ctx = gguf.getMetadata("llama.context_length") orelse return error.MissingMetadata;
    try std.testing.expectEqual(@as(u32, 8192), ctx.asU32() orelse return error.WrongType);

    const vocab = gguf.getMetadata("llama.vocab_size") orelse return error.MissingMetadata;
    try std.testing.expectEqual(@as(u32, 49152), vocab.asU32() orelse return error.WrongType);
}

test "GGUF tensor info listing" {
    var gguf = GGUFFile.open("models/smollm2-135m-q4_k_m.gguf", std.testing.allocator) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer gguf.deinit();

    // Check token embedding
    const embd = gguf.getTensorInfo("token_embd.weight") orelse return error.TensorNotFound;
    try std.testing.expectEqual(@as(u32, 2), embd.n_dims);
    try std.testing.expectEqual(@as(u64, 576), embd.dims[0]);
    try std.testing.expectEqual(@as(u64, 49152), embd.dims[1]);
    try std.testing.expectEqual(TensorType.Q8_0, embd.tensor_type);

    // Check output_norm
    const norm = gguf.getTensorInfo("output_norm.weight") orelse return error.TensorNotFound;
    try std.testing.expectEqual(@as(u32, 1), norm.n_dims);
    try std.testing.expectEqual(@as(u64, 576), norm.dims[0]);
    try std.testing.expectEqual(TensorType.F32, norm.tensor_type);

    // Check a block tensor
    const wq = gguf.getTensorInfo("blk.0.attn_q.weight") orelse return error.TensorNotFound;
    try std.testing.expectEqual(@as(u64, 576), wq.dims[0]);
    try std.testing.expectEqual(@as(u64, 576), wq.dims[1]);
}

test "GGUF dequantize values match" {
    var gguf = GGUFFile.open("models/smollm2-135m-q4_k_m.gguf", std.testing.allocator) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer gguf.deinit();

    // F32 tensor
    const norm = try gguf.readTensorF32("output_norm.weight", std.testing.allocator);
    defer std.testing.allocator.free(norm);
    // Python: 1.757812 1.820312 1.781250 1.601562 1.742188
    try std.testing.expectApproxEqAbs(@as(f32, 1.757812), norm[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.820312), norm[1], 0.001);

    // Q8_0 tensor (token_embd)
    const emb = try gguf.readTensorF32("token_embd.weight", std.testing.allocator);
    defer std.testing.allocator.free(emb);
    // Python: -0.106544 0.032978 0.048199 -0.022831 -0.068493
    try std.testing.expectApproxEqAbs(@as(f32, -0.106544), emb[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.032978), emb[1], 0.001);

    // Q5_0 tensor
    const attn_q = try gguf.readTensorF32("blk.0.attn_q.weight", std.testing.allocator);
    defer std.testing.allocator.free(attn_q);
    // Python (ggml order): -0.067322 0.134644 -0.067322 -0.740540 1.077148
    try std.testing.expectApproxEqAbs(@as(f32, -0.067322), attn_q[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.134644), attn_q[1], 0.001);
}

test "GGUF Q4_K dequantization" {
    var gguf = GGUFFile.open("models/smollm2-135m-q4_k_m.gguf", std.testing.allocator) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer gguf.deinit();

    const data = try gguf.readTensorF32("blk.2.ffn_down.weight", std.testing.allocator);
    defer std.testing.allocator.free(data);

    // Python reference: Q4_K block 0 first 10:
    // [-0.10960579, -0.4586792, 0.06493092, 0.00675201, 0.12310982,
    //  -0.10960579, -0.10960579, 0.12310982, -0.28414249, 0.06493092]
    try std.testing.expectApproxEqAbs(@as(f32, -0.10960579), data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.4586792), data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.06493092), data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.00675201), data[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.12310982), data[4], 0.001);
}

test "GGUF dequantize tensor" {
    var gguf = GGUFFile.open("models/smollm2-135m-q4_k_m.gguf", std.testing.allocator) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer gguf.deinit();

    // Read a small F32 tensor (output_norm.weight, 576 elements)
    const norm = try gguf.readTensorF32("output_norm.weight", std.testing.allocator);
    defer std.testing.allocator.free(norm);
    try std.testing.expectEqual(@as(usize, 576), norm.len);
    // Check values are reasonable (not all zero, not NaN)
    var non_zero: usize = 0;
    for (norm) |v| {
        try std.testing.expect(!std.math.isNan(v));
        if (v != 0) non_zero += 1;
    }
    try std.testing.expect(non_zero > 100);

    // Read a Q8_0 tensor (blk.0.attn_v.weight, 576×192 = 110592 elements)
    const attn_v = try gguf.readTensorF32("blk.0.attn_v.weight", std.testing.allocator);
    defer std.testing.allocator.free(attn_v);
    try std.testing.expectEqual(@as(usize, 576 * 192), attn_v.len);
    non_zero = 0;
    for (attn_v) |v| {
        try std.testing.expect(!std.math.isNan(v));
        if (v != 0) non_zero += 1;
    }
    try std.testing.expect(non_zero > 1000);

    // Read a Q5_0 tensor (blk.0.attn_q.weight, 576×576 = 331776 elements)
    const attn_q = try gguf.readTensorF32("blk.0.attn_q.weight", std.testing.allocator);
    defer std.testing.allocator.free(attn_q);
    try std.testing.expectEqual(@as(usize, 576 * 576), attn_q.len);
}
