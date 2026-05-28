# Security

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Please report via GitHub's private security advisory feature at
<https://github.com/SMC17/znap/security/advisories/new> (or the eventual public fork owner).

This includes:
- Code execution as the running user via crafted model file (.bin, .gguf, .znap).
- Buffer overrun in the SIMD / SDOT inner loops with adversarial input shape.
- Numerical correctness bugs that produce wrong actions/tokens silently.
- Tokenizer-decode out-of-bounds reads.

Include in your report: model file (or minimal reproducer), Zig version,
target architecture (M1/x86_64), reproduction command.

## Response timeline

Single-author / small-team project. Realistic:
- Acknowledgement: 72 hours.
- Triage: 14 days.
- Patch: 90 days for confirmed issues.

## Scope-of-trust

znap trusts:
- The model/weight file at face value. The user is expected to have validated the source of any .bin/.gguf/.znap before running.
- The standard Zig stdlib (no third-party Zig deps in the core micro-prediction path).
- The host CPU's documented SIMD ISA (NEON on Apple Silicon, AVX2/AVX-512 on x86_64; explicit scalar fallback present).

znap does NOT trust:
- Network sources (none queried at runtime in the core library).
- Untrusted model files (run in a sandbox if loading random .gguf or unvetted .znap).

## Coordinated disclosure

90-day clock, extendable on mutual agreement.

This file was added during the 2026-05 OSS launch polish per the PRE_LAUNCH_CHECKLIST audit.
