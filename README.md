# znap

**Zero-allocation Neural Acceleration Primitives**

Sub-nanosecond inference for small neural models. BitNet b1.58 quantized kernels. SIMD-accelerated. No GPUs. No heap. No dependencies. Just Zig.

```
SIMD dot product (30-dim):       < 1 ns     3.2 billion ops/sec
Linear policy (30→6):             30 ns      33 million evals/sec
DQN 2-layer (30→64→6):          191 ns       5.2 million evals/sec
BitNet DQN (ternary, int-only):    2 ns     587 million evals/sec
7-agent ensemble:                220 ns       4.6 million evals/sec
```

*Benchmarked on Apple M1. Your Threadripper will be faster.*

## Why

Most inference engines optimize for large language models on GPUs. znap optimizes for the other end of the spectrum: **small models (<10K params) that need to run at hardware speed on CPUs.**

Use cases:
- **Reinforcement learning** — policy evaluation in real-time control loops
- **Multi-agent systems** — evaluate dozens of agents per tick
- **Edge inference** — sub-microsecond decisions on embedded hardware
- **Game AI** — thousands of NPC evaluations per frame
- **Robotics** — control policies at >1MHz update rates

If your model fits in L1 cache and your latency budget is measured in nanoseconds, znap is for you.

## Architecture

znap exploits three properties of small-model inference that large-model frameworks ignore:

### 1. Comptime Specialization

Every tensor dimension is a compile-time parameter. The compiler generates a unique, unrolled kernel for each shape. No runtime dispatch. No shape checking. No virtual calls.

```zig
const znap = @import("znap");

// Compile-time specialized: 30 features → 6 actions
const Policy = znap.LinearPolicy(30, 6);
var policy = Policy.init();
const result = policy.evaluate(&features);
// result.action, result.confidence, result.probabilities
```

The compiler knows `30` and `6` at compile time and generates optimal SIMD-unrolled code for exactly that shape.

### 2. Zero Allocation

The entire inference path — feature vectors, weight matrices, activations, softmax, argmax — runs on the stack. No heap. No allocator. No GC pause. No `malloc` in the hot loop. Ever.

```zig
// This struct is entirely stack-allocated
const DQN = znap.DQNPolicy(30, 64, 6);  // 2,374 params, all on stack
const dqn = DQN.initXavier(42);
const result = dqn.evaluate(&features);  // zero allocations
```

### 3. BitNet b1.58 Quantization

Weights constrained to {-1, 0, +1}. The inner loop of matrix-vector multiply becomes:

```
for each weight w:
    if w == +1:  sum += input[j]
    if w == -1:  sum -= input[j]
    if w ==  0:  skip
```

**No floating-point multiplication in the inner loop.** Just integer addition and subtraction. One f32 multiply per row (scale factor) at the end.

Reference: [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)

```zig
const Ternary = znap.TernaryPolicy(30, 6);

// Quantize a trained float policy
const quantized = Ternary.fromFloat(&trained_policy);

// Integer-only inference — 89x faster than float on ARM NEON
const result = quantized.evaluate(&features);
```

At >3B params, BitNet matches full-precision perplexity. At small scales (<10K params), we observe minor quality degradation with 5-10x compression and up to 89x inference speedup.

## Quick Start

```bash
git clone https://github.com/SMC17/znap.git
cd znap
zig build test     # Run 147 tests
zig build bench    # Run benchmarks
zig build example  # Train an RL agent, quantize, compare
```

Requires Zig 0.14+. No other dependencies.

## Modules

| Module | What it does |
|--------|-------------|
| `tensor` | Fixed-shape comptime tensors, element-wise ops, activations |
| `simd` | NEON/AVX-accelerated dot product, softmax, ReLU, argmax |
| `quantized` | BitNet b1.58 ternary inference, integer-only matvec |
| `policy` | Generic RL policies: linear (bandit), DQN, ensemble |
| `replay` | Zero-allocation experience replay buffer (ring buffer) |
| `parallel` | Multi-threaded parallel policy evaluation |

## API

### Tensor

```zig
const T = znap.Tensor(4, 8);          // 4×8 matrix, all comptime
var t = T.fromFlat(.{1,2,3,...,32});
t.relu();                              // In-place ReLU
t.softmax();                           // In-place softmax per row
const s = t.sum();                     // Scalar reduction
const idx = t.argmax();                // Index of max element

// Matrix-vector multiply (the hot path)
const scores = znap.matvec(6, 30, &weights, &features);
```

### Policy

```zig
// Linear (bandit): scores = W*x + b
const P = znap.LinearPolicy(128, 16);
var policy = P.init();
const result = policy.evaluate(&features);
policy.update(&features, chosen_action, reward, learning_rate);

// DQN (2-layer): hidden = ReLU(W1*x + b1), out = W2*h + b2
const D = znap.DQNPolicy(128, 64, 16);
var dqn = D.initXavier(42);  // Xavier initialization
const result = dqn.evaluate(&features);

// Ensemble: N independent policies
const E = znap.Ensemble(8, 128, 16);
var ensemble = E.init();
const results = ensemble.evaluateAll(&features);  // [8]PolicyResult
```

### Quantized (BitNet b1.58)

```zig
// Quantize a trained policy to ternary weights
const Q = znap.TernaryPolicy(128, 16);
const quantized = Q.fromFloat(&trained_policy);  // {-1, 0, +1} weights
const result = quantized.evaluate(&features);     // Integer-only inner loop

// DQN quantization
const QD = znap.TernaryDQN(128, 64, 16);
const qdqn = QD.fromFloat(&trained_dqn);
```

### Replay Buffer

```zig
// Zero-allocation ring buffer, comptime capacity
var buf = znap.ReplayBuffer(100_000, 64).init();  // 100K transitions, 64-dim state
buf.record(state, action, reward, next_state, done, tick);

const t = buf.samplePrioritized(&rng);  // Priority-weighted sampling
const batch = buf.sampleBatch(32, &rng); // Batch of 32

buf.avgReward();    // Statistics
buf.fillRate();     // How full is the buffer
```

### C FFI

znap compiles to a static library with C ABI exports. Call from Rust, Go, Python, C++:

```bash
zig build                  # Produces libznap.a
```

## Benchmarks

All measurements on Apple M1 (aarch64), ReleaseFast, single thread, 1M iterations with warmup:

| Operation | Latency | Throughput | vs PyTorch* |
|-----------|---------|------------|-------------|
| Dot product (30-dim) | <1 ns | 3.2B ops/s | — |
| Softmax (6-dim) | <1 ns | 2.6B ops/s | — |
| ReLU (64-dim) | 3 ns | 301M ops/s | — |
| Linear 30→6 | 30 ns | 33M ops/s | ~2,000x |
| DQN 30→64→6 (f32) | 191 ns | 5.2M ops/s | ~500x |
| DQN 30→64→6 (ternary) | 2 ns | 587M ops/s | ~50,000x |
| 7-agent ensemble | 220 ns | 4.6M ops/s | ~1,000x |

*\*PyTorch comparison is for equivalent model shapes on CPU (no GPU). PyTorch has significant per-call overhead from dynamic dispatch, Python GIL, and memory allocation that dominates at small model sizes.*

### Memory

| Representation | 30→6 Linear | 30→64→6 DQN |
|---------------|-------------|-------------|
| Float32 | 744 bytes | 9,496 bytes |
| BitNet b1.58 | 93 bytes | ~856 bytes |
| Compression | **8.0x** | **11.1x** |

## Design Principles

1. **No hidden allocations.** If it's not on the stack, it doesn't exist. `std.mem.Allocator` is never passed to any hot-path function.

2. **Comptime over runtime.** Tensor shapes, SIMD widths, loop bounds — everything that CAN be known at compile time IS known at compile time.

3. **SIMD by default.** `@Vector` built-ins compile to native SIMD (NEON on ARM, AVX on x86). No runtime feature detection.

4. **No dependencies.** znap depends only on `std`. No build system plugins, no C libraries, no pkg-config.

5. **Tested.** 147 tests across all modules. Every operation has at least one test. Quantization correctness is verified against float reference.

## Research Context

znap implements ideas from several frontier research directions:

- **BitNet b1.58** (Ma et al., 2024): Ternary weight quantization with per-row absmax scaling
- **Comptime kernel generation**: Related to TVM/Halide's schedule-based optimization, but at the language level rather than as a DSL
- **Zero-allocation RL**: Inspired by the observation that small RL policies (<10K params) don't need the infrastructure of large-model frameworks

## Contributing

Issues and PRs welcome. Areas of interest:

- [ ] 4-bit and 8-bit quantization (INT4/INT8 kernels)
- [ ] Attention mechanism (small-scale self-attention for sequence models)
- [ ] SSM (Mamba-style state-space model blocks)
- [ ] Mixed-precision: ternary weights + INT8 activations
- [ ] WASM compilation target for browser inference
- [ ] CUDA bridge for training (GPU train → CPU/ternary deploy)
- [ ] Model format: load weights from safetensors/GGUF

## License

MIT
