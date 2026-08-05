#!/usr/bin/env bash
# PROFILE-CACHE: 262144 context with prefix caching, on 8x RTX 3090.
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

# F. 768 is the ceiling at max_model_len 262144 on 24 GiB cards: 1024 needs
# 0.55 GiB for admission against a ~0.89 GiB pool that must also hold ~0.6 GiB
# of prefill transients.
F="${F:-768}"
# Pool size. Must clear the admission requirement for one max-length request
# and still leave the transients room. Do not raise this much further: past
# ~1600 blocks NCCL loses the race for its lazily allocated buffers and dies
# with ncclUnhandledCudaError inside a PP broadcast, not a clean OOM.
BLOCKS="${BLOCKS:-1000}"
# Half of F, so two prefills can share one scheduler step. Free for
# single-stream because the adaptive gate only applies the cap when two or more
# prefills are queued.
LONG_PREFILL="${LONG_PREFILL:-384}"
MAXLEN="${MAXLEN:-262144}"

[[ -n "$VLLM_BIN" ]] || { echo "vllm not found; set VLLM_BIN" >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "model not found at $MODEL; set MODEL" >&2; exit 2; }
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/serve-profile-cache-$(date +%s).log"

# One log file per boot, deliberately. Appending across boots mixes tracebacks
# from different runs and makes a post-mortem read the wrong failure.
echo "log: $LOG"

# Compressor state stays in its default absolute-position placement -- that is
# what makes it prefix-addressable. Unset rather than =0 so an exported value
# from a previous PROFILE-P8 run in the same shell cannot leak in.
unset VLLM_DSV4_COMPRESSOR_WINDOWED VLLM_DSV4_COMPRESSOR_WINDOW || true

export VLLM_SM86_DCP=1                              # the SM8x DCP path
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0   # estimate reserves 1.46 GiB, real use is 0.07
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=48         # default 512; 48 measured harmless here
export VLLM_DSV4_WARMUP=1
export VLLM_DSV4_SM86_INDEXER_TILES=1
export VLLM_DSV4_DELTA_GATHER=1
export VLLM_DSV4_DELTA_GATHER_BUDGET_MB=192         # 512 fits the staging alone, but not next to
                                                    # this profile's larger per-request reservation
export VLLM_DSV4_FLASH_PREFILL=0                    # measured 2-4% slower at every length
export VLLM_DSV4_FLASH_DECODE=1                     # this is what makes decode context-flat
export VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ALONE: pairing it with
                                                          # max_split_size_mb silently disables it

# Two caps on the process-lifetime workspace arena. The arena grows to the
# largest request any caller ever makes and is then frozen, so whatever these
# pin stays pinned -- on 24 GiB cards that is the difference between ~90 MiB
# and ~290 MiB of free memory at steady state. Both default to upstream
# behaviour when unset; see envs.py for the failure modes.
#   BUFFER_TOKENS: raw tokens, floor is max_num_seqs * max_model_len. Upstream
#   uses max_model_len * 40, which pins ~331 MiB here. Raise this in lockstep
#   with --max-num-seqs below -- too low fails SILENTLY.
export VLLM_DSV4_INDEXER_PREFILL_BUFFER_TOKENS=$(( 4 * MAXLEN ))
#   CHUNK_SIZE: halves a ~260 MiB bf16 gather buffer. Correct at any value;
#   a step with more concurrent prefills than this takes an extra pass.
export VLLM_DSV4_PREFILL_CHUNK_SIZE=2

# Deliberately NOT set. vLLM auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH for
# DeepseekV4ForCausalLM, which sets CompilationMode.NONE and so turns inductor
# off; decode then launches thousands of un-fused kernels per token. Setting it
# to 0 turns inductor back on and fuses them. Whether that is a win depends
# entirely on how fast your host launches kernels, and it is not free:
#
#   host                   per-kernel launch   decode with inductor    prefill
#   Zen3 desktop, 8c       4.21 us             +4%   (51.1 -> 53.2)    -3.8%
#   Zen2 server, 64c       8.21 us             +126% (23.4 -> 52.9)    +32%
#
# The cost is ~1.2 GiB per GPU of extra weights+workspace, which on 24 GiB
# cards is mutually exclusive with this profile: the pool here is 650 blocks of
# admission reservation plus 256 blocks of 262144 context, and there is no
# 1.2 GiB to give back. Measured, it boots and then dies on a long prefill.
#
# So: leave it alone on a fast host. On a slow-launch host, uncomment it AND
# accept a shorter max_model_len or switch to launch-profile-p8.sh, whose ring
# placement reserves ~3 blocks instead of 650.
#
#   export VLLM_USE_BREAKABLE_CUDAGRAPH=0

"$VLLM_BIN" serve "$MODEL" \
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
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}' \
  --host "$HOST" --port "$PORT" 2>&1 | tee "$LOG"

# The pipeline means $? is tee's; take vllm's instead.
exit "${PIPESTATUS[0]}"
