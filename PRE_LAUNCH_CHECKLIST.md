# PRE_LAUNCH_CHECKLIST — znap

**Class:** PROMOTE (M7 P9) — Zig zero-allocation neural inference engine. **M2 (Zig lab) lane confirmed as single launch-ready Zig artifact.**
**Visibility today:** PRIVATE.
**Public-push status:** **DEFERRED** per 2026-05-12 evening doctrine.
**Audit date:** 2026-05-12.
**Audit sources:** local clone HEAD `6140c51`, M7 v3 plan I, M2 lane DONE confirmation, this lane's direct audit.

## §1. README polish

**Current state.** ~150-line README. Headline framing: *"Zero-allocation Neural Acceleration Primitives — Zig neural inference engine. Loads pretrained models. Generates text. No Python. No PyTorch. No dependencies."* Verifiable example output (106.6 tok/s on Apple M1 with stories15M).

**Specific claims:**
- Apple M1 SIMD dot product (30-dim): < 1 ns, 3.2 billion ops/s
- Linear policy (30→6): 31 ns, 33 million evals/s
- DQN 2-layer (30→64→6): 194 ns, 5.2 million evals/s
- 7-agent ensemble: 222 ns, 4.6 million evals/s
- vs NumPy/PyTorch: 34-138× speedup table
- 395 tests via `zig build test`
- BitNet b1.58 ternary weight matrices, 8× memory compression
- "Zero allocation. The entire inference hot path runs on the stack."

### Strengths (relative to other PROMOTE candidates)

- **Reproducibility built in.** Quick Start has the exact `curl` URLs for stories15M model + tokenizer + the `zig build run` invocation. A reader can literally reproduce the 106.6 tok/s claim in ~5 minutes.
- **Honest scope.** "Small-model inference" — no claim of llama.cpp parity, no enterprise framing.
- **Per-operation evidence** in the benchmark table. NumPy/PyTorch comparison is grounded in measurable operations.

### Actions

- [ ] **Evidence-pointer pass.** Each benchmark number should link to the source: e.g., `[evidence: bench/sdot_30.zig, M1 Pro 10-core, Zig 0.14.0, ReleaseFast]`. Today the numbers float without compile-flag / hardware-profile disclosure.
- [ ] **Hardware-portability claim.** "Apple M1" benches don't generalize. State explicitly: "These numbers are Apple M1 results. On x86_64 AVX-512 the SIMD path may differ; we have not yet measured." This is the Type-I downgrade.
- [ ] **Zero-allocation claim verification.** The README states "no `malloc` in the inner loop" — verify by Zig's `std.heap.GeneralPurposeAllocator` with `safety = true` showing zero allocation events in the bench harness. Add this to the test suite if not already.
- [ ] **Speedup claim methodology.** "138× faster than NumPy" — what was the NumPy comparison code? Was it `np.dot` on the same 30-dim vectors? Was BLAS-multi-threading enabled or disabled? Add the comparison code to `bench/comparison/` for reviewer access.
- [ ] **BitNet b1.58 lossless claim.** "Zero quality loss" needs a defining metric (regret? KL-divergence? action-match rate?). Today "same action" is the qualifier — define "same."
- [ ] **Add `--help` documentation block.** Per the MCP-dead doctrine, `--help` IS the discovery surface. Verify `zig build run -- --help` produces a comprehensive help text; if not, add it.
- [ ] **Cross-link to `n0theta`** (M7 P6) — the future inference-cache-layer story per M7 v5.

## §2. LICENSE

**Current state.** MIT, `Copyright (c) 2026 Sean Collins`.

**Type-I/II concern.** Same as `n0theta` — MIT violates the AGPL-for-public-OSS canon from memory `project_sovereign_edge_cloudflare_decisions`. The MIT grant is fine if irrevocable for code that has already been public-shared, but the repo is PRIVATE today, so the going-forward license can still be changed.

**However:** `znap` is positioned as **a primitive / library that other projects embed.** AGPL on a primitive forces every embedder to AGPL. That may be intentional (sovereign-stack canon says yes); it may be undesirable for adoption (most ML primitive libs are MIT/Apache).

### Actions

- [ ] **License-canon decision.** Two paths:
  - **Path A (canon-consistent):** Re-license to AGPL-3.0-or-later. Matches `sentinel-sbom` / `sovereign-edge` / `sovereign-offense-harness` / planned-AGPL `n0theta`. Limits adoption but aligns with the sovereign-stack thesis.
  - **Path B (adoption-friendly):** Stay MIT. Acknowledged break from the AGPL canon for primitive-libs. **Recommendation: Path B** if `znap` is intended to be embedded by external projects; document the rationale.
- [ ] **Add `NOTICE`** with `SPDX-License-Identifier:` line.
- [ ] **Add `SPDX-License-Identifier:`** to top of each `*.zig` source file.
- [ ] **Document the license-canon decision** in a one-line README addition: *"License: MIT (intentional — `znap` is positioned as an embeddable primitive; sovereign-stack canon is AGPL, this is the documented exception)."*

## §3. SECURITY.md via GitHub advisory routing

**Current state.** No `SECURITY.md`.

### Actions

- [ ] **Add SECURITY.md** with GH-advisory routing. Modified template for an inference engine:

```markdown
# Security

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Please report via GitHub's private security advisory feature at
<https://github.com/SMC17/znap/security/advisories/new>.

This includes:
- Code execution as the running user via crafted model file (.bin, .gguf).
- Buffer overrun in the SIMD inner loops with adversarial input shape.
- Numerical correctness bugs that produce wrong tokens silently.
- Tokenizer-decode out-of-bounds reads.

Include in your report: model file (or minimal reproducer), Zig version,
target architecture (M1/x86_64), reproduction command.

## Response timeline

Single-author project. Realistic:
- Acknowledgement: 72 hours.
- Triage: 14 days.
- Patch: 90 days for confirmed issues.

## Scope-of-trust

znap trusts:
- The model file at face value. The user is expected to have validated the
  source of the .bin/.gguf before running.
- The standard Zig stdlib (no third-party Zig deps).
- The host CPU's documented SIMD ISA (NEON on Apple Silicon, AVX2/AVX-512
  on x86_64).

znap does NOT trust:
- Network sources (none queried at runtime).
- Untrusted model files (run in a sandbox if loading random .gguf).

## Coordinated disclosure

90-day clock, extendable on mutual agreement.
```

## §4. .gitignore review

**Current state.** Minimal — 8 lines: `.zig-cache/`, `zig-out/`, `*.o`, `*.a`, `*.so`, `*.dylib`, `models/*.bin`, `models/*.gguf`.

### Actions

- [ ] **Add secrets-hardening lines** (consistent with sovereign-stack):

```
# secrets / local-only state
.env
.env.*
*.key
*.pem
*.p12
*.crt

# editor / IDE state
.idea/
.vscode/
*.swp
.DS_Store
```

- [ ] **Verify no committed cache/build artifacts.** `git ls-files | grep -E '\.zig-cache|zig-out|\.bin$|\.gguf$'` — **verified clean as of audit.**

## §5. Secret scan

**Result: CLEAN.**

### Actions

- [ ] Re-run secret scan before any public-push moment.

## §6. Evidence-level annotations on every public claim

| Claim | Evidence today | Type-I (overclaim) | Type-II (missed) | Action |
|---|---|---|---|---|
| 106.6 tok/s on stories15M | Real run in README | Apple M1 specific | x86_64 perf unknown | State hardware explicitly |
| 3.2 billion SIMD ops/s | "< 1 ns dot product" | Theoretical peak vs sustained? | Cache-warm vs cache-cold | Document measurement methodology |
| 138× faster than NumPy | Table in README | NumPy with what BLAS? threading? | Single-threaded NumPy is the easy baseline | Specify NumPy build flags |
| 34× faster than PyTorch | Table | Same — PyTorch eager mode vs scripted? | torch.compile would change numbers | Specify PyTorch mode |
| 395 tests pass | `zig build test` | Test coverage not stated | A test count without breakdown isn't quality | Document test-class breakdown |
| Zero allocation in hot path | Stated | Verifiable via Zig allocator instrumentation | "Hot path" definition? | Add allocator-trace test |
| BitNet b1.58 zero quality loss | Stated | Defined relative to what baseline? | Quality metric undisclosed | Define quality metric |
| 8× memory compression | "744 → 93 bytes" for 30→6 policy | Compression varies by input | Scaled to larger models? | State scaling rule |
| "No dependencies" | True for Zig stdlib only | Zig stdlib IS a dependency, just upstream-trusted | If someone wants reproducibility w/o Zig install... | Acknowledge Zig stdlib as upstream |

**Strongest single recommendation:** `znap`'s README is the most evidence-grounded of the 5 PROMOTE candidates. Most of the work here is *labeling* what's already present, not adding new evidence.

## §7. Ship-gate (final blockers, ordered by criticality)

### P0
- [ ] **License-canon decision** (Path A AGPL vs Path B MIT, documented).
- [ ] **Add SECURITY.md** with GH-advisory routing.
- [ ] **Evidence-pointer pass on benchmark numbers** (hardware profile + Zig optimize mode).
- [ ] **Hardware-portability disclosure** in README (M1 numbers don't generalize).

### P1
- [ ] **`.gitignore` secrets-hardening additions.**
- [ ] **Add CODE_OF_CONDUCT.md + CONTRIBUTING.md** (consistent with sovereign-stack templates).
- [ ] **Tag a release** — README quotes 106.6 tok/s but there's no `v0.1.0` git tag. Tag (annotated, SSH-signed) the current HEAD as `v0.1.0`.
- [ ] **Cross-link to `n0theta`** if/when the n0theta inference-cache integration ships.
- [ ] **Zero-allocation verification test** committed to `tests/`.

### P2
- [ ] **NumPy/PyTorch comparison code** committed to `bench/comparison/`.
- [ ] **BitNet quality metric definition** in README.
- [ ] **`--help` text completeness review.**

### Public-push gate (DEFERRED until green-light)

- [ ] (Deferred) `gh repo edit SMC17/znap --visibility public --accept-visibility-change-consequences`.
- [ ] (Deferred) Verify `gh api repos/SMC17/znap --jq '.private,.visibility'` returns `false public`.
- [ ] (Deferred) **HN Show HN** — title: *"Show HN: znap — zero-allocation neural inference in Zig, no Python, no PyTorch"*. The MCP-dead doctrine's CLI-first framing fits the HN audience perfectly here.
- [ ] (Deferred) r/zig + Zig Discord `#projects`.
- [ ] (Deferred) r/LocalLLaMA (very high-fit audience; the "no Python, no PyTorch" hook is exactly the r/LocalLLaMA pain-point).
- [ ] (Deferred) lobste.rs `programming + ai`.
- [ ] (Deferred) Cold outreach to Mitchell Hashimoto / Loris Cro / Andrew Kelley (Zig core) inviting reads.
- [ ] (Deferred) `brew tap stax/tap/znap` formula submission.

## §8. Status

**Verdict at this audit:** **CLOSEST_TO_READY of the 5 PROMOTE candidates.** Substantively launch-ready already; the 4 P0 blockers are housekeeping (license decision, SECURITY.md, evidence-pointer labels, hardware disclosure) — none require new code. Estimated wall-time to `READY_PRIVATE_AUDITED`: **1 day focused.**

This is the recommended **first-out-the-door** candidate when the no-public-push doctrine clears. The other 4 PROMOTE candidates depend on:
- n0theta → bench-reproducibility (weeks)
- Greek-Net → 6-fix list + author's-STATUS-doc reconciliation (~1.5 days)
- GR33K-N3T → prior-art sub-agent + proof-verification sub-agent + LICENSE (2-3 weeks)
- OilGame → README from scratch + LICENSE (2-3 days)

`znap` clears all 4 of those gates *today*, modulo the housekeeping pass.
