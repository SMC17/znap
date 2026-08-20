# znap status (Zig 0.16 build port)

Last check: 2026-08-20. Host Zig: **0.16.0**.

`build.zig` was migrated from `addStaticLibrary` / `addSharedLibrary` /
`root_source_file` to Zig 0.16 `addLibrary` + `createModule` / `root_module`.
The Build API no longer fails.

The package as a whole is **not compiled** on 0.16 until remaining language
errors are gone (`src/neon.zig` / `src/sdot_matmul.zig` asm clobbers,
`std.fs.cwd`, `ArrayList.init`, `std.io.fixedBufferStream`,
`std.process.getEnvVarOwned`).

Throughput / nanosecond figures in older notes were **not** re-measured on
this host. Do not treat them as current.

Keep the repository private.
