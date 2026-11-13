# Contributing to znap

- Zig 0.16.0. `zig build test --summary all` must stay green.
- A performance claim in a PR needs the command, the machine, the Zig version and the commit. Benchmarks must keep inputs opaque to the optimizer (see `clobber()` in `benchmarks/bench.zig`).
- Good first work: re-run an experiment binary and compare with paper/inverse_scaling.md; x86_64 SIMD paths; GGUF loading coverage.

MIT. Security reports: see SECURITY.md.
