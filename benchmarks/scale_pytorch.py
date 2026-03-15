#!/usr/bin/env python3
"""
Deep scaling experiment: DQN ternary quantization crossover (PyTorch).

This extends znap's deep_scaling.zig to GPU-scale models (100K → 100M params).
Run on your Threadripper + 5090 or rented A100/H100 for the full curve.

Reproduces the EXACT same experiment: train DQN on argmax task, quantize to
ternary, measure float↔ternary agreement. Combined with the Zig results
(<500K params), this maps the complete crossover curve.

Usage:
    python3 benchmarks/scale_pytorch.py [--device cuda]
    python3 benchmarks/scale_pytorch.py --device cpu  # slower but works

Requirements:
    pip install torch numpy
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def ternary_quantize(weight):
    """Quantize a weight matrix to ternary {-1, 0, +1} with per-row absmax scaling."""
    absmax = weight.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = weight / absmax
    ternary = torch.zeros_like(weight)
    ternary[normalized > 0.5] = 1.0
    ternary[normalized < -0.5] = -1.0
    return ternary, absmax

def int8_quantize(weight):
    """Quantize a weight matrix to INT8 with per-row scaling."""
    absmax = weight.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scale = absmax / 127.0
    quantized = (weight / scale).round().clamp(-127, 127)
    return quantized, scale

class DQN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim):
        super().__init__()
        self.w1 = nn.Linear(feature_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        h = torch.relu(self.w1(x))
        return self.w2(h)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

def ternary_forward(model, x):
    """Forward pass with ternary-quantized weights."""
    # Layer 1
    w1_t, s1 = ternary_quantize(model.w1.weight)
    h = torch.relu(x @ (w1_t * s1).T + model.w1.bias)
    # Layer 2
    w2_t, s2 = ternary_quantize(model.w2.weight)
    return h @ (w2_t * s2).T + model.w2.bias

def int8_forward(model, x):
    """Forward pass with INT8-quantized weights."""
    w1_q, s1 = int8_quantize(model.w1.weight)
    h = torch.relu(x @ (w1_q * s1).T + model.w1.bias)
    w2_q, s2 = int8_quantize(model.w2.weight)
    return h @ (w2_q * s2).T + model.w2.bias

def run_experiment(F, H, A, device, epochs=100, train_n=None):
    """Train a DQN on argmax task and measure quantization agreement."""
    model = DQN(F, H, A).to(device)
    params = model.param_count

    # Training data
    if train_n is None:
        train_n = min(params * 10, 50_000)

    X_train = torch.rand(train_n, F, device=device)
    targets = X_train[:, :A].argmax(dim=1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    batch_size = min(256, train_n // 4)
    for epoch in range(epochs):
        perm = torch.randperm(train_n, device=device)
        for i in range(0, train_n, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_train[idx])
            loss = criterion(logits, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_eval = torch.rand(10_000, F, device=device)
        eval_targets = X_eval[:, :A].argmax(dim=1)

        float_logits = model(X_eval)
        float_actions = float_logits.argmax(dim=1)
        float_acc = (float_actions == eval_targets).float().mean().item() * 100

        tern_logits = ternary_forward(model, X_eval)
        tern_actions = tern_logits.argmax(dim=1)
        tern_acc = (tern_actions == eval_targets).float().mean().item() * 100
        tern_agree = (float_actions == tern_actions).float().mean().item() * 100

        int8_logits = int8_forward(model, X_eval)
        int8_actions = int8_logits.argmax(dim=1)
        int8_agree = (float_actions == int8_actions).float().mean().item() * 100

        # Decision margin
        sorted_logits = float_logits.sort(dim=1, descending=True).values
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()

        # Inference timing
        start = time.perf_counter_ns()
        for _ in range(1000):
            _ = model(X_eval[:1])
        float_us = (time.perf_counter_ns() - start) / 1000 / 1000

        # Weight density
        w1_t, _ = ternary_quantize(model.w1.weight)
        w2_t, _ = ternary_quantize(model.w2.weight)
        w1_density = (w1_t != 0).float().mean().item() * 100
        w2_density = (w2_t != 0).float().mean().item() * 100

    return {
        'params': params,
        'train_n': train_n,
        'float_acc': float_acc,
        'tern_acc': tern_acc,
        'tern_agree': tern_agree,
        'int8_agree': int8_agree,
        'margin': margin,
        'float_us': float_us,
        'w1_density': w1_density,
        'w2_density': w2_density,
    }

def main():
    parser = argparse.ArgumentParser(description='DQN ternary quantization scaling experiment')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*90}")
    print(f"  znap/PyTorch — Deep Scaling: DQN Ternary Quantization Crossover")
    print(f"  Device: {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'})")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'='*90}\n")

    configs = [
        # (F, H, A, label)
        (8, 32, 4, "8x32x4"),
        (16, 64, 4, "16x64x4"),
        (32, 128, 8, "32x128x8"),
        (64, 256, 8, "64x256x8"),
        (64, 512, 8, "64x512x8"),
        (128, 512, 16, "128x512x16"),
        (128, 1024, 16, "128x1024x16"),
        (256, 1024, 16, "256x1024x16"),
        (256, 2048, 16, "256x2048x16"),
        (512, 2048, 32, "512x2048x32"),
        (512, 4096, 32, "512x4096x32"),
        (1024, 4096, 32, "1024x4096x32"),
    ]

    print(f"  {'Scale':<18} {'Params':>8} {'TrainN':>7} {'FloatAcc':>9} {'TernAcc':>8} {'TAgree':>7} {'I8Agree':>8} {'Margin':>7} {'F32us':>6} {'W1den':>6} {'W2den':>6}")
    print(f"  {'─'*18} {'─'*8} {'─'*7} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*6}")

    results = []
    for F, H, A, label in configs:
        try:
            r = run_experiment(F, H, A, device)
            results.append((label, r))
            print(f"  {label:<18} {r['params']:>8} {r['train_n']:>7} {r['float_acc']:>8.1f}% {r['tern_acc']:>7.1f}% {r['tern_agree']:>6.1f}% {r['int8_agree']:>7.1f}% {r['margin']:>7.2f} {r['float_us']:>5.1f} {r['w1_density']:>5.1f}% {r['w2_density']:>5.1f}%")
        except Exception as e:
            print(f"  {label:<18} FAILED: {e}")

    print(f"\n  {'─'*18} {'─'*8} {'─'*7} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*6}")

    # Find crossover point
    print(f"\n  CROSSOVER ANALYSIS:")
    for label, r in results:
        marker = ""
        if r['tern_agree'] >= 95:
            marker = " ← TERNARY VIABLE (>95%)"
        elif r['tern_agree'] >= 90:
            marker = " ← APPROACHING (>90%)"
        elif r['tern_agree'] >= 80:
            marker = " ← IMPROVING (>80%)"
        print(f"    {label:<18} {r['params']:>8} params → {r['tern_agree']:.1f}% ternary agree{marker}")

    print(f"\n  Combined with znap Zig results (≤500K params), this maps the complete")
    print(f"  crossover curve from toy scale to production scale.\n")

    # CSV output for plotting
    print(f"\n  CSV for plotting:")
    print(f"  params,float_acc,tern_acc,tern_agree,int8_agree,margin")
    for label, r in results:
        print(f"  {r['params']},{r['float_acc']:.1f},{r['tern_acc']:.1f},{r['tern_agree']:.1f},{r['int8_agree']:.1f},{r['margin']:.3f}")

if __name__ == '__main__':
    main()
