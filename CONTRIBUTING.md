# Contributing to znap

znap is a sovereign micro-prediction primitive (core strength: 31 ns linear/ternary policy eval + lossless compression results).

## Development

- Zig 0.14+ for exact reproduction of the published 31 ns / paper numbers (see AUDIT_NOTES/QUARANTINE_ZIG_016_BUILD.md for 0.16+ status).
- `zig build test` for the core suite.
- `zig build bench` / `zig build audit` / `zig build scaling` etc. for the paper experiments.
- Focus areas that strengthen the QM positioning: better SDOT/AVX/NEON kernels, standalone challenge improvements, zero-alloc proofs, more linear policy scales with falsifiable margin data.

## Claims & Evidence

Every load-bearing performance or correctness claim must be accompanied by a pre-registered hypothesis + falsifier (stax-experiment style or equivalent). Type I (overclaim) and Type II (missed risk) audits are non-negotiable.

## License

MIT for the primitive (intentional documented exception to sovereign-stack AGPL default for embeddable libraries). See LICENSE and the top of README.md.

Pull requests that add private workstation details, weaken reproducibility, or remove falsifiers will be closed.

## Communication

Issues and PRs on the public repo. For security, use the advisory flow in SECURITY.md.
