# znap

**Zero-allocation Neural Acceleration Primitives**

A neural inference engine written in Zig. Loads pretrained models. Generates text. No Python. No PyTorch. No dependencies.

```
$ zig build run -- models/stories15M.bin -t 0.9

 Once upon a time, there was a little bird who loved to fly in the sky.
 She flew everywhere she went, and the wind was so strong that it made
 the bird feel weak. One day, she met a wise old owl who told her about
 the importance of using her wings to fly...

 128 tokens in 1201ms (106.6 tok/s)
```

## Quick Start

```bash
git clone https://github.com/SMC17/znap.git
cd znap

# Download a pretrained model (~58MB)
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

# Generate text
zig build run -- models/stories15M.bin

# Run benchmarks
zig build bench

# Run all 395 tests
zig build test
```

Requires Zig 0.14+. No other dependencies.

## Performance

Small-model inference, benchmarked on Apple M1:

```
SIMD dot product (30-dim):       < 1 ns      3.2 billion ops/s
Linear policy (30→6):             31 ns       33 million evals/s
DQN 2-layer (30→64→6):          194 ns        5.2 million evals/s
7-agent ensemble:                222 ns        4.6 million evals/s
```

Compared to equivalent operations in Python:

| Operation | NumPy | PyTorch | znap | Speedup |
|-----------|-------|---------|------|---------|
| Linear 30→6 | 4.28 μs | 3.32 μs | **0.031 μs** | 138× |
| DQN 30→64→6 | 6.38 μs | 6.69 μs | **0.194 μs** | 34× |
| 7-agent ensemble | 29.5 μs | 21.3 μs | **0.222 μs** | 96× |

Text generation with pretrained stories15M (6 layers, 288 dim, 6 heads):
- **110 tokens/sec** on Apple M1 CPU
- Zero dependencies, single binary

## What's Inside

| Module | What it does |
|--------|-------------|
| `llama2` | Load and run pretrained LLaMA-architecture models |
| `tensor` | Fixed-shape comptime tensors, SIMD-accelerated ops |
| `simd` | NEON/AVX vector operations (dot, softmax, relu) |
| `quantized` | BitNet b1.58 ternary inference (integer-only inner loop) |
| `policy` | RL policies: linear, DQN, ensemble |
| `attention` | Multi-head attention, RoPE, KV cache, sampling |
| `transformer` | Decoder-only transformer, SwiGLU MLP |
| `gpu` | Hardware abstraction (CUDA, Metal, CPU backends) |
| `gpu_train` | Backprop training loop with margin regularization |
| `weights` | Binary weight format, load/save |
| `tokenizer` | Byte-level BPE tokenizer |
| `train` | Gradient-free training (bandit, evolution, CEM) |
| `replay` | Zero-allocation experience replay buffer |
| `parallel` | Multi-threaded parallel evaluation |

## Research

znap includes 10 benchmark experiments that produced a novel research finding:

**[Lossless Compression of Neural Policies](paper/inverse_scaling.md)** — We show that linear RL policies can be compressed 8-14× to ternary weights with **zero quality loss** (100% action agreement, KL divergence = 0.0). For DQN policies, INT8 maintains >91% agreement at all scales. Results validated on three backends (Zig SIMD, PyTorch CPU, PyTorch Metal GPU).

Run the experiments yourself:

```bash
zig build scaling        # Scaling law across 7 model sizes
zig build formal         # Formal verification of lossless compression
zig build deep-scaling   # DQN scaling to 75K params
zig build audit          # Honest benchmarks with trained weights
python3 benchmarks/scale_pytorch.py --device mps  # GPU-scale experiments
```

## Architecture

znap exploits three properties unique to Zig:

**Comptime Specialization.** Every tensor dimension is a compile-time parameter. The compiler generates a unique, SIMD-unrolled kernel for each shape. No runtime dispatch.

```zig
const znap = @import("znap");
const Policy = znap.LinearPolicy(30, 6);  // specialized at compile time
var policy = Policy.init();
const result = policy.evaluate(&features); // 31 nanoseconds
```

**Zero Allocation.** The entire inference hot path runs on the stack. No allocator, no GC, no `malloc` in the inner loop.

**BitNet b1.58.** Ternary weight matrices ({-1, 0, +1}) use integer addition instead of floating-point multiplication. 8× memory compression for linear policies with zero quality loss.

```zig
const Q = znap.TernaryPolicy(30, 6);
const compressed = Q.fromFloat(&trained_policy);  // 744 → 93 bytes
const result = compressed.evaluate(&features);     // same action, 8× smaller
```

## Benchmarks

```bash
zig build bench          # Raw speed benchmarks
zig build audit          # Honest audit with trained weights
zig build scaling        # Scaling law experiment
zig build formal         # Formal verification
zig build deep-scaling   # DQN scaling curve
zig build dqn-crossover  # Quantization crossover analysis
zig build qat            # Quantization-aware training
zig build margin         # Margin training experiment
```

## Contributing

Areas of interest:

- [ ] GGUF format loading (for HuggingFace model compatibility)
- [ ] INT4/INT8 quantized inference kernels
- [ ] CUDA backend for GPU training
- [ ] Metal compute shaders for Apple GPU inference
- [ ] Attention with sliding window + group-query (GQA)
- [ ] WASM target for browser inference
- [ ] Larger model support (stories42M, stories110M)

## License

MIT
