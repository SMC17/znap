# znap status

Last check: 2026-11-12, Zig 0.16.0, Apple M2 Max, macOS.

- `zig build test --summary all`: 53/53 steps, 631/631 tests. Proof level: **unit-tested**.
- `zig build bench -Doptimize=ReleaseFast`: see the README table. Proof level: **benchmarked** (one machine, three runs).
- `zig build run -Doptimize=ReleaseFast -- models/stories15M.bin`: 691-704 tok/s over five runs.
- All 11 binaries compile on Zig 0.16.0. The experiment outputs (audit, scaling, deep-scaling, dqn-crossover, margin, qat, formal) have not been re-run and compared with paper/inverse_scaling.md.

Retired: the 31 ns / 110 tok/s M1 numbers from Zig 0.14 builds. The M2 Max
linear-policy result (32 ns) is in the same range; it is a new measurement,
not a confirmation of the old one.
