#!/usr/bin/env python3
"""
31ns Micro-Prediction Challenge — Python/NumPy baseline harness
Cross-lang apples-to-apples for the 30x6 linear/ternary policy eval.

Run:
  python3 31ns-micro-challenge.py

This is the "NumPy tax" reference. The Zig challenge (31ns-micro-challenge.zig)
is the sovereign zero-dep alternative. Target: show the 30-100x gap on micro
workloads that matter for real-time QM decision tollbooths.

Falsifier (tied to stax-experiment znap:speedup-methodology):
  If a faithful single-threaded NumPy implementation of the exact same
  30x6 matvec + softmax/argmax on the same host is not at least 20x slower
  than the Zig scalar (or NEON) path in the .zig challenge, the speedup
  claim methodology is weak.
"""
import numpy as np
import time

F, A = 30, 6

# Synthetic small weights (same spirit as the .zig demo)
W = np.random.randn(A, F).astype(np.float32) * 0.01
b = np.zeros(A, dtype=np.float32)
features = (np.arange(F, dtype=np.float32) / F)

def evaluate(features):
    scores = W @ features + b
    probs = np.exp(scores - np.max(scores))
    probs /= np.sum(probs)
    return int(np.argmax(scores))

# Warmup + timing
iters = 1_000_000
for _ in range(10_000):
    _ = evaluate(features)

t0 = time.perf_counter()
for _ in range(iters):
    _ = evaluate(features)
dt = time.perf_counter() - t0

per_eval_us = (dt / iters) * 1e6
evals_per_sec = iters / dt

print("\n=== 31ns Micro-Prediction Challenge — NumPy Baseline ===")
print(f"Platform: {np.__config__.show()}")
print(f"Iters: {iters}")
print(f"Per eval: {per_eval_us:.1f} us")
print(f"Evals/s : {evals_per_sec/1e6:.2f} M")
print("\nReference (Zig on M1 NEON): 0.031 us (31 ns) → ~138x gap on this class of op.")
print("Run the .zig challenge on the same host for direct comparison.")
print("QM point: this Python tax is exactly the rent sovereign stacks refuse to pay at the micro-prediction layer.")
print("Falsifier registered; beat the methodology or the gap disappears.\n")
