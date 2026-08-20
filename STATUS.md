# znap status (Zig 0.16)

Last check: 2026-08-20. Host Zig: **0.16.0**.
Experiment: `exp-1787256490-513747286` (verdict: **refuted** — package does not compile as a whole).

`build.zig` uses Zig 0.16 `addLibrary` + `root_module`. Remaining failures are
language/stdlib, not the Build API.

Historical 31 ns / tok/s numbers on `main` were **not** re-measured on this
host. They are not a current proof. Keep the repository private.
