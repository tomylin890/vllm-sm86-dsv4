# deploy/

Everything needed to reproduce the numbers in the top-level README, rather than
take them on trust.

## Launching

```bash
MODEL=/path/to/DeepSeek-V4-Flash-0731 \
VLLM_BIN=/path/to/venv/bin/vllm \
./launch-profile-cache.sh
```

Two profiles, and they are mutually exclusive by construction — the engine
refuses the wrong combination at startup rather than degrading quietly:

| | `launch-profile-cache.sh` | `launch-profile-p8.sh` |
|---|---|---|
| prefix caching | yes | no |
| max-num-batched-tokens | 768 | 1024 |
| num-gpu-blocks-override | 1000 | 650 |
| long-prefill-token-threshold | 384 | 512 |
| compressor state | absolute position | ring (position modulo 512) |
| suits | agents, multi-turn, RAG | batch, one-shot long documents |

Both default to `max_model_len` 262144, TP4 + PP2 + dcp4, and fp8 KV. Every
flag is explained in [docs/install.md](../docs-dsv4/install.md); the reasoning
behind the two profiles is in the top-level README.

After boot, send one request at each length you care about before you measure
anything. Triton compiles per chunk-count bucket on first sight, so the first
request at a new length runs at roughly half speed and that number is not real.

## Verifying

`verify/` holds the harnesses these results were produced with. They are stdlib
only (no requests, no numpy), take `--base-url`, and write machine-readable
JSON next to a human-readable table.

| script | what it answers |
|---|---|
| `p11_prefill_probe.py` | cold prefill throughput at several lengths; each point uses a fresh filler seed so nothing hits the cache |
| `p11_maxctx.py` | does a near-max-context request work, cold and warm |
| `p11_determinism.py` | same prompt N times: is the stack reproducible run to run |
| `p11_t1_equality.py` | cache hit vs full recompute, token identity and logprob deltas |
| `p11_t2_needle.py` | needle-in-a-haystack across contexts and depths, cold vs warm per cell |
| `p11_t3_agent_bench.py` | multi-turn agent sessions: per-turn TTFT and the exact cache-hit length |
| `p11_t4_eviction.py` | concurrent long sessions under pool pressure, parsing the per-group hit-length instrumentation |
| `p11_conc_probe.py` | samples the scheduler while two requests are in flight: are they really concurrent |
| `p11_pl_ab.py` | power-limit A/B against a standing server; restores the limit on exit |
| `p11_compare_t1.py` | offline comparison of two `p11_t1_equality.py` runs |

Example:

```bash
cd verify
python3 p11_prefill_probe.py 16384 65536 131072 200704 253952
python3 p11_maxctx.py 262080
python3 p11_t3_agent_bench.py run --config B --out t3.json
```

Two things worth knowing before you read your own results:

`p11_t2_needle.py` and the agent benchmark drive the model with thinking
disabled. With reasoning on, a 64-token budget is spent entirely inside the
think trace and the answer never appears, which reads as a failure when nothing
failed.

The harnesses that run long (`p11_t2_needle.py`, `p11_t4_eviction.py`) pace
themselves in rounds with a cooldown that waits both a fixed time and for the
GPUs to return to their pre-round temperature. On eight cards in one chassis
the wall you hit first is power delivery and heat, not compute.
