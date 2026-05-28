# Zig for Serious Micro-Prediction — 31ns Ternary Edge Inference Under Quantitative Mercantilism

**Evidence-first launch of znap: the zero-allocation, comptime-specialized Zig inference primitive that evaluates linear RL policies in 31 nanoseconds on Apple M1 with NEON SDOT, delivers lossless 8-14× ternary compression for linear policies, and gives sovereign operators a real alternative to the Python tax at the decision tollbooth.**

## The 31 ns Result (with full Type I disclosures)

On Apple M1 (aarch64 + NEON SDOT asm path, Zig ReleaseFast ~0.14):

- Linear policy (30 features → 6 actions): **31 ns**, 33 million evals/s
- 7-agent ensemble: 222 ns
- Speedup vs equivalent NumPy/PyTorch CPU ops on the same hardware: **34–138×**

**Critical disclosures (read the AUDIT_NOTES/ in the package before citing):**
- These numbers are Apple M1 hardware-verified. On x86_64 Linux the kernels take the explicit scalar fallback and are slower. Reproduce the 31 ns bar only on equivalent Apple Silicon class hardware.
- Build system in the tree is 0.14-era; does not compile on Zig 0.16+ without porting (see quarantine note). The core policy + sdot + quantized logic is the portable artifact.
- All claims pre-registered in the stax-experiment register with explicit falsifiers before any public artifact was written.

The source of the speed is not magic: comptime tensor dimensions + hand-written NEON SDOT (4×16 MACs per instruction) + zero heap allocations on the hot path + 2-bit packed ternary weights that turn the inner loop into integer add/sub.

## Lossless Ternary Compression (the inverse scaling paper)

Trained linear RL policies can be compressed post-training to ternary weights ({-1, 0, +1}) with **100% action agreement** and **KL divergence = 0.0000** versus the float32 reference — at every scale tested (10 to 4,112 parameters, 8–14× memory reduction).

The mechanism is not that ternary is inherently safe. It is that gradient descent on cross-entropy / bandit objectives implicitly maximizes decision margins. The gap between the top action and the runner-up grows during training (observed margin ~2.61 at 30×6 after 50k steps). That margin absorbs the quantization perturbation.

For 2-layer DQN policies, ternary eventually fails at scale, but INT8 post-training quantization stays above 91% agreement from 420 to 4.3M parameters. The paper gives the complete deployment recipe.

Paper: `paper/inverse_scaling.md` in the package (or the original mac-mining mirror). All Zig experiments: single `zig build` commands. PyTorch scale sweeps included for cross-validation.

## Quantitative Mercantilism Framing

Quantitative Mercantilism studies tollbooths in the topology of flows — policy, inventory, physical infrastructure. The model layer is rapidly commoditizing into a utility. Value is migrating to the appliance layer: sovereign runtimes, deterministic orchestration, introspectable surfaces, zero external rent.

Real-time micro-prediction and policy execution (signals → action in <1 µs) is a tollbooth. Every trading system, every venue matching engine, every robotics control loop that can run its decision policy locally with nanosecond predictability and no Python interpreter or cloud call owns that position.

znap is the reference primitive that proves the Zig substrate can seize it. 31 ns linear policy eval with zero dependencies and lossless 8–14× compression is not a benchmark curiosity. It is the concrete evidence that first-principles engineers can build the next layer of sovereign infrastructure on top of it instead of renting it.

This is the material for the "graveyard for non-Zig tooling" at the micro scale. Superior, auditable, zero-rent performance that compounds when embedded.

## The Benchmark Challenge (reproduce or refute)

In the package: `benchmarks/31ns-micro-challenge.zig` + `.py`

- Standalone timing of the 30×6 ternary policy path (comptime-style, zero-alloc pattern).
- Cross-lang NumPy baseline for direct apples-to-apples.
- Explicit falsifier tied to the pre-registered stax-experiment claims.
- Target: match or beat 31 ns on M1-class aarch64 + Zig 0.14 ReleaseFast (scalar on everything else; bring your own NEON/AVX intrinsics).

Run it. Time it on your hardware. Fork the kernels. Beat the reference. Then publish your numbers with the same falsifier discipline.

## Call to Army-Builders

If you are a first-principles engineer who believes:
- Sovereign stacks beat rented ones,
- Evidence (nanoseconds + compression tables + reproducible builds) beats narrative,
- Zig + comptime + explicit SIMD is the correct substrate for the appliance layer,

then this primitive is for you.

Embed it in your trading signal engine. Use the ternary policy as the policy head for a larger sovereign model. Port the SDOT pattern to your target ISA. Wire it into stax-experiment or your own verification harness. Extend the lossless result to new architectures.

The package is MIT (documented exception for embeddable primitives). The reconnaissance clone, paper, and full audit notes are in the mac-mining lineage. The clean tree is at [public location when pushed].

Fork. Time. Build. Measure. Register your own claims with falsifiers.

The tollbooth is open. The receipts are the 31 ns numbers and the zero in the KL column.

— Inward leverage phase, 2026-05-26

**Links (when public):**  
znap clean package + challenge + paper + AUDIT_NOTES/  
stax-experiment register (search for znap:* lanes)  
Original reconnaissance + full mac-mining context  

All load-bearing claims were pre-registered with falsifiers before this post was drafted. Read the register. Attack the claims with evidence.
