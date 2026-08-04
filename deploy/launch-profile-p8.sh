#!/usr/bin/env bash
# PROFILE-P8: maximum throughput, NO prefix caching, on 8x RTX 3090.
#
# This is the configuration the README's measured numbers come from. Every
# value in it is a measured boundary rather than a round number -- see
# docs/install.md for what each flag buys and docs/troubleshooting.md for what
# happens when one of them is wrong.
#
# Override anything with an environment variable:
#   MODEL=/path/to/DeepSeek-V4-Flash-0731 VLLM_BIN=/path/to/venv/bin/vllm ./launch-profile-cache.sh
set -euo pipefail

MODEL="${MODEL:-$HOME/models/DeepSeek-V4-Flash-0731}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# F. The ring frees enough per-request reservation that 1024 fits here.
F="${F:-1024}"
# Pool size. Smaller than PROFILE-CACHE's because the ring shrinks what each
# request must reserve; two concurrent requests at moderate context fit.
BLOCKS="${BLOCKS:-650}"
# Half of F, so two prefills can share one scheduler step. This is a necessary
# condition for healthy concurrency -- with threshold == F one prefill consumes
# the whole per-step budget and the second request cannot even queue behind it.
LONG_PREFILL="${LONG_PREFILL:-512}"
MAXLEN="${MAXLEN:-262144}"

[[ -n "$VLLM_BIN" ]] || { echo "vllm not found; set VLLM_BIN" >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "model not found at $MODEL; set MODEL" >&2; exit 2; }
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/serve-profile-p8-$(date +%s).log"

# One log file per boot, deliberately. Appending across boots mixes tracebacks
# from different runs and makes a post-mortem read the wrong failure.
echo "log: $LOG"

# The ring. Addressing compressor state by absolute position MODULO a window
# pins the per-request reservation at a constant (~3 blocks) instead of O(F),
# which is what lets F reach 1024 and what makes concurrency work. It is also
# why prefix caching cannot be used here: a position-modulo layout has no
# relationship to a token prefix. The engine refuses the combination loudly.
export VLLM_DSV4_COMPRESSOR_WINDOWED=1
export VLLM_DSV4_COMPRESSOR_WINDOW=512   # the default; spelled out for clarity

export VLLM_SM86_DCP=1                              # the SM8x DCP path
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0   # estimate reserves 1.46 GiB, real use is 0.07
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=48         # default 512; 48 measured harmless here
export VLLM_DSV4_WARMUP=1
export VLLM_DSV4_SM86_INDEXER_TILES=1
export VLLM_DSV4_DELTA_GATHER=1
export VLLM_DSV4_DELTA_GATHER_BUDGET_MB=512         # the ring leaves room for the full budget
export VLLM_DSV4_FLASH_PREFILL=0                    # measured 2-4% slower at every length
export VLLM_DSV4_FLASH_DECODE=1                     # this is what makes decode context-flat
export VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ALONE: pairing it with
                                                          # max_split_size_mb silently disables it

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name dsv4-flash-0731 --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --decode-context-parallel-size 4 --dcp-comm-backend a2a \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --no-enable-flashinfer-autotune \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 --max-num-batched-tokens "$F" \
  --long-prefill-token-threshold "$LONG_PREFILL" \
  --num-gpu-blocks-override "$BLOCKS" \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}' \
  --host "$HOST" --port "$PORT" 2>&1 | tee "$LOG"
