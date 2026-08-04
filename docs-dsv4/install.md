## Installation

### Before you start

Hardware and driver requirements are in the "Environment" section of README.md; listed here are only the items that will block the install:

- Eight SM86 cards, with the TP group not spanning a PCIe switch. Every single number in the two configurations below was measured for this eight-card 24GB layout.
- driver 590.48, and torch 2.13.0+cu130 inside the venv.
- Python 3.12. `pyproject.toml` declares `>=3.10,<3.15`; 3.12 is what upstream documents and what this machine actually runs.
- nvcc. Only the flash-mla step needs it, and its CUDA version has to match the version torch in the venv reports (`python -c "import torch; print(torch.version.cuda)"`, 13.0 on this machine). vLLM itself is not compiled — see the next section.
- Disk: the model is about 156GB in fp8 (P0 recorded 167GB pulling it from HF at the time, about 20 minutes), plus the precompiled wheel and the flash-mla build output; leave 200GB or more.
- The model `deepseek-ai/DeepSeek-V4-Flash-0731` itself — pull it to the machine however you normally do, since the launch commands below take a local path.

Do not set `PYTHONOPTIMIZE`. `python -O` strips asserts out entirely, and several geometry checks are exactly what fail closed via assert: the "stride vs write length" contract in KV block zeroing (`vllm/v1/worker/utils.py`), and the two dcp-related assertions in the hybrid coordinator (group type, block size divisibility, which is P11's B1/B2). Once they are stripped, an invalid configuration does not blow up; it quietly computes the wrong thing, or quietly boots. Refuse to start at all when `PYTHONOPTIMIZE` has any value, for this reason.

### Installing this branch

The wheel's base is upstream main at `62195e9784ebec1ece42b88a861734e0702cc2d5`. Relative to it, this clone's HEAD changes 63 files (`git diff --name-only 62195e978..HEAD`), all of them `.py` — `csrc/`, `CMakeLists.txt` and `cmake/` are untouched, not one line. 32 of those 63 come from haosdent's DSV4 SM8x support (`f8ea5bb16`, also pure Python); the remaining 43 are this branch. So the entire C++/CUDA extension can be reused straight from that wheel, and vLLM itself does not need to be compiled on this machine — that would cost an hour at minimum.

Put differently, what makes `VLLM_USE_PRECOMPILED` legitimate here is "nobody touched a compilation unit", not "the diff is small". The first thing to do after a rebase is to rerun that `git diff` and confirm there are still no non-`.py` files; otherwise the `.so` extracted from the wheel and the Python in the tree are not the same build.

Fetch the matching wheel to the machine first. `VLLM_PRECOMPILED_WHEEL_LOCATION` takes a URL or a local path (`setup.py` checks `os.path.isfile` first). Keeping it local means a reinstall does not download again, and which wheel got installed when stays visible:

```bash
mkdir -p ~/dsv4-dcp/wheels
curl -L -o ~/dsv4-dcp/wheels/vllm-0.26.1rc1.dev227+g62195e978-cp38-abi3-manylinux_2_28_x86_64.whl \
  'https://wheels.vllm.ai/62195e9784ebec1ece42b88a861734e0702cc2d5/vllm-0.26.1rc1.dev227%2Bg62195e978-cp38-abi3-manylinux_2_28_x86_64.whl'
```

The venv is built with uv, without `--seed`, so there is no `pip` module inside it — every install after this has to go through `uv pip install --python <the venv's python>`, and `python -m pip` will simply say `No module named pip`. (The build hints in `flash_mla_prefill.py` / `flash_mla_decode.py` in the tree say `python -m pip`; that assumes a seeded venv, and copying it verbatim fails.)

```bash
uv venv --python 3.12 ~/dsv4-dcp/venv

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=~/dsv4-dcp/wheels/vllm-0.26.1rc1.dev227+g62195e978-cp38-abi3-manylinux_2_28_x86_64.whl
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.26.1rc1.dev227+g62195e978

uv pip install --python ~/dsv4-dcp/venv/bin/python -e ~/dsv4-dcp/vllm --torch-backend=auto
```

`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM` is not optional. This clone has none of upstream's version tags, only the project's own phase tags (`P4`/`P6`/`P7`/`P8`, and they are lightweight tags — without `--tags`, `git describe` tells you outright there is no annotated tag). What `git describe --tags` gives is `P8-12-gdc5487ef2`, which is not PEP440, so setuptools-scm cannot derive a version number. The dist-scoped variable is used (`vllm` normalizes to `VLLM`) rather than the unscoped `SETUPTOOLS_SCM_PRETEND_VERSION` because the latter would apply to every subsequent setuptools-scm build in the same shell.

The value is just the base wheel's version string. After the install, `uv pip show vllm` reports `0.26.1rc1.dev227+g62195e978.precompiled` — the `.precompiled` suffix is added by `setup.py` itself when it takes the precompiled path (`vllm.__version__` does not carry it, because `_version.py` is written before the suffix is appended). Seeing it means the `.so` really did come from the wheel and not from a local compile.

If the branch is ever rebased onto a different upstream commit, the wheel URL and this version string have to change together: the `.so` is extracted from the wheel, and it must share a base with the Python in the tree.

### flash-mla sm86

`VLLM_DSV4_FLASH_DECODE=1` (on in both configurations) needs the patched fork's `fwd_sparse_decode_mla_partial` op, which upstream flash-mla does not have. This kernel is the reason decode stays flat across the full context. Without it, the warmup catalog item blows up at boot, and the exception message carries the build commands directly.

```bash
cd ~/dsv4-dcp/flash-mla-int
git checkout dcp-sm86-patches
git submodule update --init --recursive        # csrc/cutlass

~/dsv4-dcp/venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
FLASH_MLA_CUDA_ARCHS=86 uv pip install --python ~/dsv4-dcp/venv/bin/python \
  -v --no-build-isolation .

~/dsv4-dcp/venv/bin/python - <<'PY'
import torch, flash_mla
print(hasattr(torch.ops.flash_mla, "fwd_sparse_decode_mla_partial"))
PY
```

`flash-mla-int` is a clone of `https://github.com/AppMana/forks-flash-mla-int`, and `dcp-sm86-patches` is the branch this build is made from. If the tree is not on the machine yet, clone that remote to `~/dsv4-dcp/flash-mla-int` before running the commands above.

Three things are worth explaining. `FLASH_MLA_CUDA_ARCHS=86` pins nvcc to `sm_86`; that repo defaults to `80`, and what the default produces does not reach the native path on a 3090. `--no-build-isolation` is required: this extension has to be compiled against the ABI of the torch in the venv, and an isolated build environment pulls its own torch, so the `.so` it produces only blows up when it is loaded. The torch version check is deliberately placed before the build rather than at boot: the extension uses the torch-stable ABI and supports torch>=2.9, so anything older in the venv should stop you right there instead of surfacing at the first decode.

`--no-build-isolation` requires `setuptools` and `wheel` in the target venv. The vLLM install in the previous section does not guarantee they come along; if they are missing, add them with `uv pip install --python ~/dsv4-dcp/venv/bin/python setuptools wheel`.

Compile time is longer than upstream flash-mla's: in the decode translation unit each split kernel is instantiated twice (`kPartial`) and the mma kernel four times (`kFusedCombine`×`kPartial`).

### Launching

The two configurations differ in exactly five places: F, `--num-gpu-blocks-override`, `--long-prefill-token-threshold`, the prefix caching switch, and the two ring environment variables. Everything else is identical. For why there are two configurations rather than one switch, see "Choosing between the two profiles" in README.md.

The engine blocks the wrong combination: with ring on and prefix caching also on, `compressor.py` refuses to start outright and spells out in the message how each configuration should be set. This conflict is not resolved silently, because either resolution moves KV footprint by several GB.

The shared environment block:

```bash
export VLLM_SM86_DCP=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export VLLM_DSV4_WARMUP=1
export VLLM_DSV4_SM86_INDEXER_TILES=1
export VLLM_DSV4_DELTA_GATHER=1
export VLLM_DSV4_DELTA_GATHER_BUDGET_MB=192
export VLLM_DSV4_FLASH_PREFILL=0
export VLLM_DSV4_FLASH_DECODE=1
export VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| Variable | Why |
|---|---|
| `VLLM_SM86_DCP=1` | The master gate for every DCP change in this branch. With it off you get stock behavior, and dcp>1 is blocked by upstream `mla/indexer.py` with `NotImplementedError: DCP is not supported with sparse indexer KV compression`. |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | The estimate reserves 1.46GiB; measured use is 0.07GiB. The single largest win; the reasoning is in "Memory traps on 24GB cards" in README.md. |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | Cap on the indexer's logits scratch, default 512. Everything dropping to 64 saves is transient, and no side effects were measured. |
| `VLLM_DSV4_WARMUP=1` | On by default; it is written out because it is the "do not pay Triton JIT inside a measured request" item: after the weights are loaded and the cache is bound, a dummy batch runs each resolved sparse MLA prefill/decode kernel family once. It deliberately goes through the real production path and the real cache — Triton specialization keys not only on constexprs but also on the alignment properties of cache strides and metadata shapes, so a specialization compiled against fake buffers does not match and the compile is wasted. |
| `VLLM_DSV4_SM86_INDEXER_TILES=1` | The fp8 MQA logits kernel switches to the consumer-branch SM86 tile configuration instead of the A100 autotune sweep. Only tiling and pipelining change; the reduction block dimensions are left alone, so the accumulation chain is unchanged. It also removes the 2-config autotune benchmark, which was one of the sources of boot-to-boot nondeterminism. Active only on compute capability 8.6. |
| `VLLM_DSV4_DELTA_GATHER=1` | Each prefill chunk gathers only the compressed entries newly completed since the previous chunk, instead of repacking and re-all-gathering the whole prefix. A compressed entry is written once at a block boundary and never changes afterwards, so the staged bytes stay valid. |
| `VLLM_DSV4_DELTA_GATHER_BUDGET_MB=192` | Per-worker total budget for those staging buffers. A (request, layer) that exceeds the budget is simply not tracked and falls back to a full gather — the degradation is per request and per layer, not a global switch. The budget table in P7-NOTES worked out 512 as "a single 256k sequence fits exactly", but that only accounts for staging itself; under PROFILE-CACHE ring is off and the per-request sliding window reservation is much larger, and in practice it has to come down to 192 for the 204800 warmup not to OOM. That trades about 1% of gather efficiency for transient headroom; see the memory section of troubleshooting.md. |
| `VLLM_DSV4_FLASH_PREFILL=0` | Prefill stays on the Triton pipeline and does not use flash-mla's fused op. |
| `VLLM_DSV4_FLASH_DECODE=1` | Decode uses the patched fork's partial op, which returns this rank's pre-sink output plus the natural-log LSE and hands it to the merge in `dcp.py` (the sink is folded exactly once, at the global max). The build in the previous section exists for this. |
| `VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1` | `--long-prefill-token-threshold` applies only when two or more prefills are queued; a single stream keeps full F-sized chunks. It is free for single stream (measured 4,646 vs 4,631, inside the noise). |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Use it alone. Combined with `max_split_size_mb`, expandable is silently disabled, with no warning of any kind. |

The debugging-only `VLLM_SM86_DET_TOPK=1` is very slow; turn it on only when a top-k tie race has to be pinned down for an A/B. The reasoning is in "No guarantee of run-to-run reproducibility" in README.md.

#### PROFILE-CACHE

```bash
unset VLLM_DSV4_COMPRESSOR_WINDOWED VLLM_DSV4_COMPRESSOR_WINDOW

~/dsv4-dcp/venv/bin/vllm serve <model path> \
  --served-model-name dsv4-flash-0731 --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --decode-context-parallel-size 4 \
  --dcp-comm-backend a2a \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 4 \
  --long-prefill-token-threshold 384 \
  --num-gpu-blocks-override 1000 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}'
```

The two ring variables have to be `unset` rather than set to `0`: if the previous run was PROFILE-P8, the exports in the shell carry over into this one.

#### PROFILE-P8

```bash
export VLLM_DSV4_COMPRESSOR_WINDOWED=1
export VLLM_DSV4_COMPRESSOR_WINDOW=512
```

The serve command is identical to the one above except for these four:

```
  --max-num-batched-tokens 1024
  --long-prefill-token-threshold 512
  --num-gpu-blocks-override 650
  --no-enable-prefix-caching
```

`VLLM_DSV4_COMPRESSOR_WINDOW` must be a positive multiple of 128 and strictly greater than 128 (128 is the largest compressor lookback window, and also the block table's token alignment); the engine checks this. Memory is linear in W, and the sub-chunk count is about `ceil(F/(W-128))`; W>=F lets the compressor forward keep a single launch pair per layer. 512 is the value used in the measurements.

#### What each flag does

| Flag | Why |
|---|---|
| `--served-model-name dsv4-flash-0731` | The model name seen through the API, decoupled from the on-disk path, so moving the path does not require touching clients. |
| `--trust-remote-code` | DSV4's config and tokenizer carry remote code. |
| `--kv-cache-dtype fp8` | KV stored as fp8. Less than 1GB per card is left over after the 156GB of weights, and a bf16 KV cache for 262K does not fit. |
| `--block-size 256` | Tokens per KV block. Upward, this value determines the scheduler's block granularity; it also determines that prefix caching's hit length is `floor(previous length/1024)×1024` (1024 = block_size×dcp). |
| `--host 0.0.0.0 --port 8000` | Bind on the LAN; bound to localhost only, a request from another machine looks like "the service is up but unreachable". |
| `--tensor-parallel-size 4` | Keeps the TP group inside a single PCIe switch. TP8 hits a wall somewhere around 52-62k of context, precisely because TP's collectives cross the switch. |
| `--pipeline-parallel-size 2` | The cut across switches is placed on PP, whose traffic is an order of magnitude smaller than TP's. |
| `--decode-context-parallel-size 4` | Compressed KV is round-robined across 4 ranks along the sequence dimension; the sliding window groups are not sharded and are replicated on every rank. |
| `--dcp-comm-backend a2a` | The default `ag_rs` is 3 NCCL calls per layer; `a2a` exchanges partial outputs and LSE and then merges with Triton, down to 2. Only meaningful for MLA models, and it requires dcp>1. |
| `--no-enable-flashinfer-autotune` | Turns off the FlashInfer autotune that runs during the kernel warmup phase. On this machine it is pure boot cost. |
| `--tokenizer-mode deepseek_v4` | DSV4's own tokenizer. |
| `--tool-call-parser deepseek_v4` `--enable-auto-tool-choice` `--reasoning-parser deepseek_v4` | Parsers for the tool call and reasoning fields. If you are not serving Agent traffic these three can be dropped, with no effect on performance. |
| `--max-model-len 262144` | The context ceiling. This value goes straight into the memory budget, and lowering it is one of the three real levers. |
| `--max-num-batched-tokens` | The token budget per scheduling step, that is, F throughout this document. It determines the prefill chunk size, it is the main throughput knob, and it also drives the activation term in the memory budget. |
| `--max-num-seqs 4` | Ceiling on how many sequences can be scheduled at once. At 262144 a long request can only run single stream (see the concurrency section in README.md), and 4 leaves room for the genuine concurrency that exists below 10-20k; it is also the reason the capture sizes above stop at 4. |
| `--long-prefill-token-threshold` | Ceiling on the tokens one prefill can take per step. It must be <= half of F, otherwise a single prefill eats the whole budget and the second one never even gets the chance to queue in the same step. |
| `--num-gpu-blocks-override` | Sets the KV pool's block count directly, overriding what the profiler computed. This is what decides whether the engine boots at all: too large and it fails while NCCL allocates communicator buffers (not a clean OOM), too small and it hits the admission threshold. |
| `--gpu-memory-utilization 0.92` | Only affects the number that the override replaces; it does not appear in actual VRAM use. It is kept so the profiler's log numbers still mean something. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | The core difference between the two configurations. For this model the branch defaults to on; both sides write it out explicitly so the launch command is self-describing, and a future flip of the default will not silently change the configuration. |
| `--compilation-config` | `FULL_DECODE_ONLY` captures CUDA graphs for decode batches only. `FULL_AND_PIECEWISE` OOMs at 256K, by about 2MiB. Capture sizes stop at [1,2,4] because `--max-num-seqs 4` already caps batch size at 4, and capturing beyond that just eats VRAM for nothing. |

### Warm up after boot

A freshly booted engine must not be put under traffic, and must not be measured either. Triton JIT has to compile once for every new chunk-count bucket, roughly 7-11.5 seconds per new length, and that cost is paid again on every restart. The engine's built-in warmup catalog (`VLLM_DSV4_WARMUP=1`) handles the kernel families themselves, but the shape buckets that go over HTTP are only reached by actually sending requests.

The SOP is one request at each of five representative lengths after boot:

```
2048  16384  65536  131072  204800
```

Any failed warmup request should abort the run rather than continue, because the numbers measured from that boot are not comparable. When measuring by hand, sweep twice and take the second pass; the first pass comes out half as fast, and that is not real performance.

Under PROFILE-CACHE there is one more thing: the warmup itself fills the cache. Any later measurement that needs cold numbers has to use entirely fresh prompt content (the rack kit's T-series scripts salt the prompt) or another restart, otherwise what you measure is hits.

Startup is complete when two things hold at the same time: `Route: /v1/models` appears in the log, and `GET /v1/models` really returns 200. Either one alone misleads — the route being mounted does not mean the engine can accept requests, and a refused connection before the route is mounted does not mean startup failed.

### Reproducing the measured numbers

`deploy/` holds the two launch scripts above as runnable files, plus `deploy/verify/`,
the harnesses every number in the README was produced with. See
[deploy/README.md](../deploy/README.md) for which harness answers which question.

```bash
MODEL=/path/to/DeepSeek-V4-Flash-0731 VLLM_BIN=/path/to/venv/bin/vllm \
  deploy/launch-profile-cache.sh
```

Before you boot, make sure no stale engine is holding VRAM: `pkill -9 -f "vllm serve"`,
then poll `nvidia-smi --query-gpu=memory.used --format=csv,noheader` until every card is
back under 500MiB. A boot that profiles memory against a polluted available figure sizes
its KV pool wrong and fails later, in a place that looks unrelated.

