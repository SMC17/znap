# znap

Zero-allocation neural inference in Zig 0.16: small RL policies in tens of
nanoseconds, and llama2.c-format text generation. No Python, no PyTorch, no
dependencies. MIT.

```
$ zig build run -Doptimize=ReleaseFast -- models/stories15M.bin

 Lily was miserable. She did not want to play with her toys or read her
 books. She only wanted to go to the store with her mom...

  256 tokens in 364ms (703.6 tok/s)
```

## Status

**Unit-tested, benchmarked on one machine.** `zig build test --summary all`:
625/625 on Zig 0.16.0. Numbers below were measured on 2026-11-12, Apple M2 Max,
macOS, Zig 0.16.0, ReleaseFast, with the commands shown. They are one machine,
not a general claim; run them on yours.


## Quick Start

```bash
git clone https://github.com/SMC17/znap.git
cd znap

# Download a pretrained model (~58MB)
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

# Generate text
zig build run -Doptimize=ReleaseFast -- models/stories15M.bin

# Run benchmarks
zig build bench

# Run unit tests
zig build test
```

Requires Zig 0.16. No other dependencies.

## Performance

`zig build bench -Doptimize=ReleaseFast` (inputs are clobbered every call so
nothing is constant-folded; three runs agreed within 3%):

| Operation | Time per eval | Evals/s |
|---|---:|---:|
| Linear policy 30→6, f32 | 32 ns | 31.7 M |
| DQN 30→64→6, f32 | 212 ns | 4.7 M |
| 7-agent linear ensemble | 234 ns | 4.3 M |
| Ternary (1.58-bit) linear 30→6 | 82 ns | 12.1 M |
| Ternary DQN 30→64→6 | 820 ns | 1.2 M |

Ternary weights are 8x smaller (93 bytes vs 744 for the linear policy) but
slower than f32 on this CPU: they buy memory, not speed.

Text generation, stories15M (6 layers, dim 288), 256 tokens, five runs:
691-704 tok/s.

Older README numbers (31 ns on M1, 110 tok/s, NumPy/PyTorch ratios) were
measured with Zig 0.14 and are superseded by the table above. The Python
comparison harnesses are in `benchmarks/` and have not been rerun.

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
const result = policy.evaluate(&features);
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
zig build bench          # Speed benchmarks (table above)
zig build audit          # Audit with trained weights
zig build scaling        # Scaling across 7 model sizes
zig build deep-scaling   # DQN ternary crossover, 100 to 500K params (long run)
zig build dqn-crossover  # Quantization crossover analysis
zig build qat            # Quantization-aware training
zig build margin         # Margin training -> quantization robustness
zig build formal         # Formal verification
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
