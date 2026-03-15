#!/usr/bin/env python3
"""
Benchmark comparison: znap (Zig) vs PyTorch vs NumPy.

Runs the EXACT same operations (30→6 linear, 30→64→6 DQN) in each
framework and measures wall-clock time per operation.

Usage:
    python3 benchmarks/compare_pytorch.py

Prerequisites:
    pip install torch numpy
"""

import time
import numpy as np

FEATURE_DIM = 30
HIDDEN_DIM = 64
ACTION_DIM = 6
WARMUP = 10_000
ITERS = 1_000_000

def benchmark(name, func, iters=ITERS, warmup=WARMUP):
    """Benchmark a function, return ns/iter."""
    for _ in range(warmup):
        func()

    start = time.perf_counter_ns()
    for _ in range(iters):
        func()
    elapsed_ns = time.perf_counter_ns() - start

    ns_per_iter = elapsed_ns / iters
    us_per_iter = ns_per_iter / 1000
    ops_per_sec = 1e9 / ns_per_iter if ns_per_iter > 0 else float('inf')

    print(f"  {name:<45} {us_per_iter:>8.3f} us  {ops_per_sec:>12,.0f} ops/s")
    return ns_per_iter


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


print()
print("=" * 66)
print("  znap vs PyTorch vs NumPy — Inference Benchmark")
print("  Exact same operations, same shapes, same data")
print("=" * 66)
print()

# ── Setup ─────────────────────────────────────────────────────────

features = np.random.randn(FEATURE_DIM).astype(np.float32)
w_linear = np.random.randn(ACTION_DIM, FEATURE_DIM).astype(np.float32)
b_linear = np.random.randn(ACTION_DIM).astype(np.float32)
w1 = np.random.randn(HIDDEN_DIM, FEATURE_DIM).astype(np.float32)
b1 = np.random.randn(HIDDEN_DIM).astype(np.float32)
w2 = np.random.randn(ACTION_DIM, HIDDEN_DIM).astype(np.float32)
b2 = np.random.randn(ACTION_DIM).astype(np.float32)

# ── NumPy ─────────────────────────────────────────────────────────

print("NumPy (CPU):")

benchmark("dot product (30-dim)", lambda: np.dot(features, features))

benchmark("linear 30x6 (matmul + bias + softmax)",
          lambda: softmax(w_linear @ features + b_linear))

def numpy_dqn():
    h = np.maximum(0, w1 @ features + b1)  # ReLU
    return softmax(w2 @ h + b2)

benchmark("DQN 30x64x6 (2-layer + relu + softmax)", numpy_dqn)

def numpy_ensemble():
    for _ in range(7):
        softmax(w_linear @ features + b_linear)

benchmark("ensemble 7x linear", numpy_ensemble, iters=100_000)

# ── PyTorch ───────────────────────────────────────────────────────

print()

try:
    import torch
    torch.set_num_threads(1)  # Fair single-thread comparison

    print(f"PyTorch {torch.__version__} (CPU, 1 thread):")

    t_features = torch.from_numpy(features)
    t_w_linear = torch.from_numpy(w_linear)
    t_b_linear = torch.from_numpy(b_linear)
    t_w1 = torch.from_numpy(w1)
    t_b1 = torch.from_numpy(b1)
    t_w2 = torch.from_numpy(w2)
    t_b2 = torch.from_numpy(b2)

    benchmark("dot product (30-dim)",
              lambda: torch.dot(t_features, t_features))

    benchmark("linear 30x6 (matmul + bias + softmax)",
              lambda: torch.softmax(t_w_linear @ t_features + t_b_linear, 0))

    def torch_dqn():
        h = torch.relu(t_w1 @ t_features + t_b1)
        return torch.softmax(t_w2 @ h + t_b2, 0)

    benchmark("DQN 30x64x6 (2-layer + relu + softmax)", torch_dqn)

    def torch_ensemble():
        for _ in range(7):
            torch.softmax(t_w_linear @ t_features + t_b_linear, 0)

    benchmark("ensemble 7x linear", torch_ensemble, iters=100_000)

except ImportError:
    print("  PyTorch not installed — skipping")

# ── Reference znap numbers ────────────────────────────────────────

print()
print("znap reference (from `zig build bench`):")
print("  dot product (30-dim)                             <0.001 us    3,200,000,000 ops/s")
print("  linear 30x6 (bandit)                              0.030 us       33,000,000 ops/s")
print("  DQN 30x64x6 (2-layer)                             0.191 us        5,200,000 ops/s")
print("  ternary DQN 30x64x6 (BitNet)                      0.002 us      550,000,000 ops/s")
print("  ensemble 7x linear                                 0.220 us        4,600,000 ops/s")

print()
print("=" * 66)
print("  Run `zig build bench` separately for znap numbers on YOUR hardware.")
print("  The comparison is apples-to-apples: same shapes, same ops.")
print("=" * 66)
print()
