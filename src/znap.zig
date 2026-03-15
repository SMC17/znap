// znap — Zero-allocation Neural Acceleration Primitives
//
// A comptime-specialized inference engine for small neural models.
// Sub-nanosecond policy evaluation. BitNet 1.58-bit quantized kernels.
// No GPUs. No heap. No dependencies. Just Zig.
//
// Modules:
//   tensor    — Fixed-shape tensors with comptime dimensions
//   simd      — SIMD-accelerated vector operations (NEON/AVX)
//   quantized — BitNet b1.58 ternary inference (integer-only inner loop)
//   policy    — RL policy evaluation (bandit, DQN, ensemble)
//   replay    — Zero-allocation experience replay buffer
//   parallel  — Multi-threaded parallel inference

pub const tensor = @import("tensor.zig");
pub const simd = @import("simd.zig");
pub const quantized = @import("quantized.zig");
pub const policy = @import("policy.zig");
pub const replay = @import("replay.zig");
pub const parallel = @import("parallel.zig");
pub const attention = @import("attention.zig");
pub const transformer = @import("transformer.zig");
pub const mla = @import("mla.zig");
pub const moe = @import("moe.zig");
pub const speculative = @import("speculative.zig");
pub const gqa = @import("gqa.zig");
pub const ssm = @import("ssm.zig");
pub const weights = @import("weights.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const train = @import("train.zig");
pub const gpu = @import("gpu.zig");
pub const gpu_train = @import("gpu_train.zig");
pub const llama2 = @import("llama2.zig");
pub const gguf = @import("gguf.zig");
pub const quant_matmul = @import("quant_matmul.zig");
pub const int8_matmul = @import("int8_matmul.zig");

// Re-export core types
pub const Tensor = tensor.Tensor;
pub const Vec = tensor.Vec;
pub const matmul = tensor.matmul;
pub const matvec = tensor.matvec;

pub const TernaryMatrix = quantized.TernaryMatrix;
pub const TernaryPolicy = quantized.TernaryPolicy;
pub const TernaryDQN = quantized.TernaryDQN;

pub const LinearPolicy = policy.LinearPolicy;
pub const DQNPolicy = policy.DQNPolicy;
pub const Ensemble = policy.Ensemble;
pub const PolicyResult = policy.PolicyResult;
pub const ActionType = policy.ActionType;

pub const ReplayBuffer = replay.ReplayBuffer;
pub const Transition = replay.Transition;

pub const evaluateParallel = parallel.evaluateParallel;
pub const ParallelResult = parallel.ParallelResult;

// Transformer
pub const CausalAttention = attention.CausalAttention;
pub const MultiHeadAttention = attention.MultiHeadAttention;
pub const KVCache = attention.KVCache;
pub const Transformer = transformer.Transformer;
pub const TransformerBlock = transformer.TransformerBlock;
pub const SwiGLU = transformer.SwiGLU;
pub const rmsNorm = attention.rmsNorm;
pub const applyRoPE = attention.applyRoPE;
pub const topKSample = attention.topKSample;
pub const greedySample = attention.greedySample;

// Multi-Head Latent Attention (DeepSeek-V3)
pub const MultiHeadLatentAttention = mla.MultiHeadLatentAttention;
pub const LatentKVCache = mla.LatentKVCache;

// Mixture of Experts (DeepSeek-style)
pub const MoELayer = moe.MoELayer;
pub const MoEExpert = moe.Expert;

// Speculative Decoding
pub const SpeculativeDecoder = speculative.SpeculativeDecoder;

// Group-Query Attention + Sliding Window (Qwen-style)
pub const GroupQueryAttention = gqa.GroupQueryAttention;
pub const GQAKVCache = gqa.GQAKVCache;

// Weights I/O
pub const loadLinearPolicy = weights.loadLinearPolicy;
pub const loadDQNPolicy = weights.loadDQNPolicy;
pub const loadTransformer = weights.loadTransformer;
pub const loadTensorFromSlice = weights.loadTensorFromSlice;
pub const saveLinearPolicy = weights.saveLinearPolicy;
pub const saveDQNPolicy = weights.saveDQNPolicy;

// ── Library metadata ─────────────────────────────────────────────

pub const version = "0.1.0";
pub const version_major = 0;
pub const version_minor = 1;
pub const version_patch = 0;

test {
    _ = tensor;
    _ = simd;
    _ = quantized;
    _ = policy;
    _ = replay;
    _ = parallel;
    _ = attention;
    _ = transformer;
    _ = mla;
    _ = moe;
    _ = speculative;
    _ = gqa;
    _ = ssm;
    _ = weights;
    _ = tokenizer;
    _ = train;
    _ = gpu;
    _ = gpu_train;
    _ = gguf;
    _ = quant_matmul;
    _ = int8_matmul;
}
