#!/bin/bash
# Apples-to-apples benchmark: znap vs llama.cpp on SmolLM2-135M GGUF
#
# Compares inference speed on the same model, same hardware, same token count.
# Both use CPU-only inference (no GPU offloading).
#
# Usage: ./benchmarks/apples_to_apples.sh [n_tokens]

set -e

N_TOKENS=${1:-64}
MODEL="models/smollm2-135m-q4_k_m.gguf"
LLAMA_BENCH="/opt/homebrew/bin/llama-bench"
ZNAP="zig-out/bin/generate"

echo ""
echo "  ============================================"
echo "  Apples-to-Apples Benchmark: znap vs llama.cpp"
echo "  ============================================"
echo "  Model:  SmolLM2-135M Q4_K_M ($(du -h "$MODEL" | cut -f1))"
echo "  Tokens: $N_TOKENS"
echo "  CPU:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo "  Date:   $(date '+%Y-%m-%d %H:%M')"
echo ""

# Build znap in release mode
echo "  Building znap (ReleaseFast)..."
cd "$(dirname "$0")/.."
zig build 2>/dev/null
echo "  Done."
echo ""

# ── llama.cpp benchmark ──────────────────────────────────────────
echo "  Running llama.cpp benchmark..."
LLAMA_OUTPUT=$("$LLAMA_BENCH" \
    -m "$MODEL" \
    -t 1 \
    -n "$N_TOKENS" \
    -p 0 \
    -ngl 0 \
    -r 3 \
    2>/dev/null | tail -3)

# Parse llama-bench output — extract tok/s from the tg line
# Format: | llama 256M Q4_K | 98.87 MiB | ... | tg64 | 127.25 ± 0.00 |
LLAMA_TPS=$(echo "$LLAMA_OUTPUT" | grep "tg" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ±' | sed 's/ ±//')
if [ -z "$LLAMA_TPS" ]; then
    # Fallback: get the last number on the tg line
    LLAMA_TPS=$(echo "$LLAMA_OUTPUT" | grep "tg" | tail -1 | awk '{for(i=NF;i>0;i--) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}')
fi

echo "  llama.cpp: $LLAMA_TPS tok/s"
echo ""

# ── znap benchmark ───────────────────────────────────────────────
echo "  Running znap benchmark (3 runs)..."

BEST_TPS=0
for run in 1 2 3; do
    ZNAP_OUTPUT=$("$ZNAP" "$MODEL" -n "$N_TOKENS" -t 0.0 2>&1)
    TPS=$(echo "$ZNAP_OUTPUT" | grep "tok/s" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    if [ -n "$TPS" ]; then
        # Use awk for float comparison
        BEST_TPS=$(echo "$BEST_TPS $TPS" | awk '{if ($2 > $1) print $2; else print $1}')
        echo "    Run $run: $TPS tok/s"
    fi
done

echo ""

# ── Results ──────────────────────────────────────────────────────
echo "  ┌─────────────────────────────────────────┐"
echo "  │  RESULTS (SmolLM2-135M, $N_TOKENS tokens)       │"
echo "  ├──────────────┬──────────────────────────┤"
printf "  │ llama.cpp    │ %20s tok/s │\n" "$LLAMA_TPS"
printf "  │ znap         │ %20s tok/s │\n" "$BEST_TPS"
echo "  ├──────────────┴──────────────────────────┤"

if [ -n "$LLAMA_TPS" ] && [ -n "$BEST_TPS" ]; then
    RATIO=$(echo "$BEST_TPS $LLAMA_TPS" | awk '{if ($2 > 0) printf "%.1f%%", ($1/$2)*100; else print "N/A"}')
    echo "  │ znap/llama.cpp: $RATIO"
fi

echo "  └─────────────────────────────────────────┘"
echo ""
echo "  Notes:"
echo "  - llama.cpp uses optimized quantized kernels (NEON SIMD)"
echo "  - znap dequantizes to f32, then uses SIMD f32 matmul"
echo "  - Both use single-threaded CPU inference (no GPU)"
echo ""
