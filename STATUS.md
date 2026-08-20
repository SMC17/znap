# znap status (Zig 0.16)

Last check: 2026-08-20. Host Zig: **0.16.0**.
Experiment: `exp-1787257292-956995154` (verdict: **confirmed**).

`zig build test --summary all` on this tree: **51/51 steps succeeded;
617/617 tests passed**. Proof level: **unit-tested** for the package test
step on this host/toolchain.

What was ported beyond the Build API: inline-asm clobber structs, cwd file
I/O via `std.Io`, ArrayList `.empty`, in-memory weight streams, `std.enums.fromInt`,
and CUDA env detection without `getEnvVarOwned`.

Historical 31 ns / tok/s figures were **not** re-measured. They are
historical / unverified, not a current proof. Keep the repository private.
