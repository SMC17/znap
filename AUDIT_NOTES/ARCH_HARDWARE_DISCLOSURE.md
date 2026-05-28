# ARCH + HARDWARE DISCLOSURE NOTE (2026-05-26)
Core 31ns claim source: Apple M1 (aarch64) + NEON SDOT inline asm in src/neon.zig + src/sdot_matmul.zig (explicit `if (builtin.cpu.arch != .aarch64)` scalar fallback to plain i32 multiply-accumulate loops).
Host for this inward work: Linux x86_64 (uname -m confirmed), no NEON, no Apple Silicon.
Measured on this host: Will use scalar fallback path only. Expect 10-50x slower than M1 NEON for the same 30-dim dot / policy eval (typical for lack of 4x SDOT throughput).
For the clean OSS package + benchmark challenge:
- The 31ns / 33M evals/s numbers are **Apple M1 hardware-verified** (original mac-mining evidence).
- All public claims, README, paper copy, community post, and challenge MUST state: "31 ns linear policy (30→6) on Apple M1 with NEON SDOT; scalar fallback on x86_64 / other arches is slower. Reproduce the reference number only on equivalent aarch64 + Zig 0.14 ReleaseFast."
- Benchmark challenge artifact will include: (1) timing of the policy eval on whatever host it runs, (2) explicit reference to the M1 number as the target to beat or match, (3) cross-lang Python harness for apples-to-apples.
- This is the correct Type I discipline: the number is real on the hardware it was measured; it is not a universal "Zig is always 31ns" claim.
PRE_LAUNCH_CHECKLIST item addressed: "Hardware-portability claim. 'Apple M1' benches don't generalize."
No private workstation details exposed.
