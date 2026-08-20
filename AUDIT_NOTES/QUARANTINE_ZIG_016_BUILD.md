# QUARANTINE NOTE — Zig 0.16 Build Incompatibility (2026-05-26)
Source: /home/stax/mac-mining/gh-iceberg/clones/znap/build.zig (and transitively all steps)
Issue: Uses pre-0.15/0.16 Build API (`b.addStaticLibrary(.{...})` without proper deref in current std.Build).
Observed on host: Linux x86_64, Zig 0.16.0 (from nix or system).
Impact on claims:
- "Reproducible with a single `zig build` command" (README + paper) is true only on Zig 0.14.x (the version used for original M1 31ns measurements).
- On modern Zig 0.16 the project requires a build.zig port (small mechanical change to `b.addStaticLibrary(.{...})` or `b.addLibrary(.{.linkage=..., .root_module=...})` pattern).
Type classification:
- Type I risk: Claiming "just zig build" without version pin misleads adopters on current Zig.
- Type II risk: The core 31ns policy/sdot/quantized logic itself is portable; only the build orchestration is dated.
Disposition for clean OSS package:
- Do NOT mutate the reconnaissance clone.
- In the extracted clean tree: Pin Zig version in README / build.zig.zon or .github (Zig 0.14.0 for exact 31ns repro).
- Provide a minimal build.zig patch in NOTES/ or as a branch note.
- The micro-prediction primitive (LinearPolicy/Ternary + sdot kernels) can be vendored or extracted into a tiny single-file demo for the benchmark challenge that avoids the full build system.
- Future work (not in this cut): Port build.zig to 0.16+ while keeping comptime specialization and zero-alloc guarantees.
Evidence: Direct `zig build` failure log captured in session (addStaticLibrary error + reference trace to std/Build.zig).
Preserved: Original clone + prebuilt test_* NEON binaries remain authoritative for the M1 numbers.
This note is part of the sovereign audit trail for the znap OSS launch.

## Update 2026-08-20

`build.zig` plus remaining language/stdlib call sites were ported. On Zig
0.16.0, `zig build test --summary all` is 51/51 steps, 617/617 tests.
Historical 31 ns figures were not re-measured. See STATUS.md.
