# DeepSeek-V4-Flash-0731 on 8× RTX 3090

*[繁體中文版](README.zh-TW.md)*

284B-parameter MoE (13B active), eight consumer 24GB cards, 262144 context, prefill peaking at 5730 tok/s, decode flat at 50 tok/s across the whole context, prefix caching usable.

This is a fork of vLLM whose core is adding decode context parallelism for Ampere (SM86). Upstream DCP depends on kernels that require Hopper or newer, and DSV4's hybrid KV layout (compressed KV + sparse indexer + sliding window) was already incompatible with vLLM's context parallelism machinery. The two problems stack on top of each other, which is why there has been no usable long-context option on consumer cards.

This is a personal project, done on my own hardware. No support, and no guarantee it reproduces in another environment.

## Documentation

This README is the overview and the measured results. The details live in three documents:

- [Installation and startup](docs-dsv4/install.md) — environment requirements, how to install this fork from prebuilt wheels, building flash-mla for sm86, the full startup commands for both profiles (one line of explanation per flag), and the warmup procedure after boot
- [Architecture](docs-dsv4/architecture.md) — what DSV4's KV layout actually looks like, why vLLM's context parallelism cannot be used as-is, and what each phase of this fork did. Start here if you want to take over the code
- [Troubleshooting](docs-dsv4/troubleshooting.md) — symptom driven. Every entry is something this project actually hit, with the log line or command that identifies it


## Why this is hard

The weights alone take 156GB in fp8, against 192GB across eight cards. What is left has to hold the 262K KV cache, which comes to under 1GB per card in practice. At that scale any waste is fatal.

Consumer cards have no NVLink. This machine has two PCIe switches in a chain with four cards on each, and cross-switch bandwidth is a clear bottleneck — TP8 hits a wall between 52k and 62k of context, because the tensor-parallel collectives cross the switch. The final configuration is TP4+PP2+dcp4: TP completes inside a single switch, PP crosses the switch (an order of magnitude less traffic), and DCP shards the compressed KV.

Under a different topology this configuration is not necessarily the best one, but "do not let a TP group cross a switch" should be general.

## Measured results

Single-stream sequential submission, temperature 0, measured after warmup (why warmup is necessary is covered below).

### Profile one: PROFILE-P8, maximum throughput, no prefix caching

| | |
|---|---|
| Max context | 262144 |
| Prefill peak | 5730 tok/s (@21k) |
| Prefill average | 5248 tok/s (512→123k full sweep) |
| Decode | 49.9-51.2 tok/s, flat across the context |
| Concurrency 2, aggregate prefill | 5425 tok/s |
| Concurrency 2, aggregate decode | 63.5-80.0 tok/s |
| 200k needle retrieval | passes in 44 seconds |

The flat decode is worth explaining. Decode speed in a typical implementation degrades as the context grows; here it barely moves from short contexts all the way to 254k. The reason is that once the flash-mla sm86 sparse kernel is in place, decode cost depends only on the 512 selected tokens and not on the total sequence length.

### Profile two: PROFILE-CACHE, prefix caching enabled

| Context | Cold TTFT | Hit TTFT | Speedup |
|---|---|---|---|
| 32768 | 8.5s | 0.41s | 21× |
| 65536 | 17.3s | 0.48s | 36× |
| 131072 | 34.6s | 0.66s | 52× |
| 200704 | 56.2s | 0.86s | 65× |
| 253952 | 73.8s | 0.99s | 75× |

The cold numbers above were measured at F=512. F can be raised to 768 (see the memory accounting in the next section), and cold prefill recovers accordingly:

| Context | F=512 | F=768 | Delta |
|---|---|---|---|
| 65536 | 3788 tok/s | 4775 tok/s | +26% |
| 131072 | 3788 tok/s | 4479 tok/s | +18% |
| 200704 | 3571 tok/s | 4108 tok/s | +15% |
| 253952 | 3441 tok/s | 3839 tok/s | +12% |

So the cold-start cost of turning the cache on is around 15%, not thirty.

Multi-turn agent scenario, 5 turns per session with 200-token replies (measured at F=512; at F=768 the first turn is about 12% faster and the later turns are unaffected):

| Context | Turn 1 | Turns 2-5 | Session total TTFT (cached / uncached) |
|---|---|---|---|
| 8192 | 2.50s | 0.21-0.37s | 3.6s / 11.7s |
| 32768 | 6.55s | 0.23-0.25s | 7.5s / 44.2s |
| 65536 | 17.03s | 0.30-0.46s | 18.6s / 87.7s |
| 131072 | 34.11s | 0.36-0.42s | 35.7s / 179.8s |
| 200704 | 55.45s | 0.59-0.85s | 58.4s / 287.6s |

The hit length is exact: the `cached_tokens` reported on every turn equals `floor(previous request total length / 1024) × 1024`, with no deviation across 40 measurements. 1024 is the scheduler block size under dcp=4; the tail that does not fill a block is recomputed, which is almost imperceptible in a multi-turn conversation.

### Retrieval correctness

Needle-in-a-haystack, 5 context lengths × 8 depths, 40 cells total:

| | Accuracy |
|---|---|
| No-cache control | 39/40 |
| Cache on, cold | 38/40 |
| Cache on, hit | 38/40, identical to cold cell for cell |

Identical cell for cell is the key point: it is not only that the totals match, but that the same two cells fail and fail in the same way, so the cache did not change model behavior. The miss the two have in common sits at 253952, depth 0.2, which is the real retrieval limit at the 256K edge and has nothing to do with the cache.

Re-validated on the current head (per-group zeroing plus the arena reclaim, running the production trim below): the same 40-cell grid passes 40/40 cold and 40/40 warm, zero mismatches, this time including the 253952/0.2 cell. That cell sits at the 256K edge and flips between runs — read 38-40 as its honest band rather than reading an improvement into one sample. A concurrent variant was also run: two sessions (100k and 160k, distinct codewords) submitted together so the pool over-subscribes and blocks recycle across requests mid-flight; both needles retrieve correctly with no cross-contamination.

## Choosing between the two profiles

There are two ways to place the compressor state, and they are mutually exclusive:

| | PROFILE-P8 | PROFILE-CACHE |
|---|---|---|
| Compressor state | Ring buffer (position modulo) | Absolute position |
| prefix caching | Unavailable | Available |
| max-num-batched-tokens | 1024 | 768 |
| num-gpu-blocks-override | 650 | 1000 |
| long-prefill-token-threshold | 512 | 384 |
| Cold prefill | ~5200 tok/s | ~4100-4800 tok/s |
| Suited to | Batch work, one-shot long documents | Agents, multi-turn conversation, RAG |

The ceiling on F under PROFILE-CACHE is 768. That is a measured boundary, not a conservative pick. The admission requirement the engine reports for itself is: a single 262144 request needs 0.34 GiB at F=512, about 0.45 GiB at F=768, and 0.55 GiB at F=1024; meanwhile the physically available pool shrinks as F rises (activations grow), leaving about 0.89 GiB at F=1024. On top of that the 204800 warmup needs roughly 0.6 GiB of transient headroom. Put the three numbers side by side and F=1024 has no solution (0.55 + 0.6 = 1.15 > 0.89), while F=768 just clears.

Incidentally, `--gpu-memory-utilization` is no help here. As long as `--num-gpu-blocks-override` is set below the computed value, the pool size is decided by the override; util only affects the number that gets overridden, and it does not appear anywhere in actual VRAM usage. There are only three real levers: shrink the pool (which runs into the admission threshold), cut transients (delta gather budget, logits cap, capture sizes, max-num-seqs), or lower max-model-len.

The reason the two are mutually exclusive is that the ring placement addresses by "absolute position modulo the window", a layout with no correspondence to token prefixes, so a block retrieved by a cache hit means nothing once it is dropped into the ring. And the ring placement is exactly what compresses the per-request reservation from 858 down to 262, which is what lets F go to 1024. To get caching you have to give up the ring, and the ceiling on F drops from 1024 to 768 along with it.

For agents that trade is worth taking. A 5-turn session at 200k pays 57 seconds on every turn without the cache, 287 seconds in total; with the cache only the first turn pays (about 11 seconds more because of the F downgrade), and the remaining four turns come to under 3 seconds. That is over 150 seconds saved on a single session, and the gap widens the longer the session runs.

If the workload is one-shot long document processing, PROFILE-P8 is the answer. The cache has nothing to work with, and F=1024 throughput is the thing that matters.

The tables above are the shipped launch script's values. The reference machine's production now runs a further trim on top of PROFILE-CACHE: `--num-gpu-blocks-override 1010`, `--max-num-seqs 2`, delta-gather budget 128 MB, and `VLLM_PP_LAYER_PARTITION=21,22` — the default split hands the extra layer to PP0, which is already the tighter stage; mirroring it evens the free-memory floor. Together with the workspace-arena reclaim (331 → 130 MB, see the memory notes in `docs-dsv4/`), the floor under sustained 262144 load goes from single-digit MiB to roughly 850 MiB free, measured after a full-length request. Prefill under this trim at a 200W power limit: 4,033 / 3,968 / 3,679 / 3,385 tok/s at 16k / 65k / 131k / 200k, which is the power table's 200W column within noise.

## Environment

- 8× RTX 3090 (24GB, SM86), no NVLink
- Two PCIe switches in a chain, four cards each. The TP group has to land inside one switch
- Ryzen 5700X / 62GB RAM / Ubuntu 24.04 / driver 590.48
- PyTorch 2.13.0+cu130
- Model: DeepSeek-V4-Flash-0731, fp8, about 156GB

16 threads on the 5700X feeding eight workers is tight. In eager mode you can watch clear scheduling stragglers (one card's utilization randomly dropping to the floor); enabling CUDA graph improves this a lot, but a platform with a weaker CPU may find its bottleneck here.

Power delivery and cooling are practical limits, not theoretical ones. With eight cards at sustained full load in a consumer case, the wall you hit first is power headroom and heat, not compute — and that is easy to underestimate during planning. Two things work in practice: run stress tests in rounds (say 5 minutes a round, waiting between rounds for GPU temperature to come back to baseline), and bring the power limit down (see the power-limit tuning below: at 200W the eight cards shed 1200W and prefill loses under twenty percent).

## Power-limit tuning: 200W is the sweet spot

The stock power limit on a 3090 is 350W, so 2800W across eight of them at full load. This machine hits power before it hits compute, so the tuning is worth doing properly. I swept 350W down to 200W, comparing like for like across the same 7 lengths:

| PL | Peak prefill | 7-point mean | vs 350W | Marginal slope | Throughput per watt | Decode |
|---|---|---|---|---|---|---|
| 350W (stock) | 5,023 | 4,617 | — | — | 1.65 | 50.09 |
| 280W | 4,812 | 4,423 | -4.2% | 0.21 %/% | 1.98 | 49.85 |
| 250W | 4,645 | 4,278 | -7.3% | 0.31 %/% | 2.14 | 49.87 |
| 220W | 4,383 | 4,036 | -12.6% | 0.47 %/% | 2.29 | 49.93 |
| **200W** | **4,129** | **3,800** | **-17.7%** | **0.64 %/%** | **2.38** | **49.88** |

The "marginal slope" is how many percent of throughput you pay for each 1% of power you remove. It steepens from 0.21 all the way to 0.64, and 200W is the knee: one step further down costs 6.5-7% for 10% of power, while the gain in throughput per watt has already converged from +0.33 per step to +0.08. The performance knee and the efficiency knee land in the same place.

**Decode never moved** (50.09→49.88, all five power points between 49.8 and 50.1). That is not a coincidence: decode at batch 1 is memory bandwidth and latency bound, and GDDR6X bandwidth does not scale with core power. So across the whole curve we were only ever shaving core clocks, never touching the one number users feel most. It also gives a very usable stop signal — **if decode starts dropping at some step, the memory side is beginning to be constrained, and you should not go any lower no matter how much prefill is left**.

Converted into actual use, the cost is this: on a 200k session, the first turn's prefill goes from about 49 seconds to about 59 seconds; from the second turn on it runs off the cache, and the effect on hit TTFT is at the 0.1 second level (a hit only recomputes the trailing 1024 tokens, and most of that time is fixed overhead). What you get back is 1,200W shed across eight cards. For agent traffic dominated by cache hits, the trade is essentially free.

The power limit does not survive a reboot, and this machine happens to be in the "overheat trips the power, cold boot follows" pattern — one trip puts it back at 350W, the next stress test trips it again, and it becomes a loop. So it has to be made persistent:

```bash
# /etc/default/nvidia-powerlimit
NVIDIA_POWER_LIMIT_W=200
```

```ini
# /etc/systemd/system/nvidia-powerlimit.service
[Unit]
Description=Apply NVIDIA persistence mode and per-card power limit
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/nvidia-powerlimit.sh

[Install]
WantedBy=multi-user.target
```

`nvidia-powerlimit.sh` does three things: polls until `nvidia-smi -L` is ready (boot race condition), turns on persistence mode (otherwise the driver unloads when the last CUDA context exits and the limit is reverted), and applies the power limit. The wait loop has to live in the script rather than in the unit's `ExecStartPre` — systemd does not parse `$(...)`, so putting it in the unit gets the entire unit rejected, and the error only says "bad unit file setting" without telling you which line.

When you want to run limit numbers, unlock it by hand with `sudo nvidia-smi -pl 350`; a reboot takes it back to 200W automatically.


## Memory traps on 24GB cards

Every item below came out of measurement, and several took more than one reboot to confirm.

**VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0**. The estimate reserves 1.46GiB; the measured use is 0.07GiB. The single largest win.

**VLLM_SPARSE_INDEXER_MAX_LOGITS_MB**. Default is 512; dropping it to 48-64 has no side effects.

**`--num-gpu-blocks-override` and the `available` in the error message are not the same thing.** The available that the admission check reports is the capacity after the override has been applied; for the physical capacity you want the `Available KV cache memory` line in the log. I misread this twice, believing I was out of memory when in fact I had set the pool too small myself.

**PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True has to be used on its own.** Combined with max_split_size_mb, expandable is silently disabled, with no warning of any kind.

**Every reboot carries a warmup cost.** Triton JIT has to compile once for each new chunk-count bucket, roughly 7-11.5 seconds per new length. The deployment SOP is to run the warmup script after boot (one request at each of 5 representative lengths) before taking traffic; when measuring for yourself, sweep twice and take the second sweep. The first sweep's numbers come out half as high, and that is not real performance.

## Known behaviors

### No guarantee of run-to-run reproducibility

The same input, temperature 0, submitted sequentially, can produce different output on two runs. This has nothing to do with prefix caching — it happens with the cache off as well, and diverges earlier.

The root cause is in upstream vLLM. The DSV4 sparse indexer score is `sum_h w_h · ReLU(q_h · k)`, so any entry that matches no attention head scores exactly 0.0. k for the top-k is 512, and when there are fewer than 512 positively scoring entries, the tail of the selection is drawn from an enormous pool of exact-0.0 ties; the upstream top-k kernel hands out output slots with atomicAdd, so which tied entry wins depends on the arrival order of the atomics.

Measured (cache off, same prompt, run twice serially): the first divergence appears at generated token 9 / 0 / 2 / 1, corresponding to prompt lengths of 8192 / 32768 / 131072 / 200704 respectively.

The effect on quality is expected to be small but not zero. What gets swapped out are entries the indexer itself scored as completely irrelevant (score 0), but they still occupy one of the 512 slots and their KV still enters the attention sum. Measured needle accuracy is unaffected.

The effect on reproducibility, on the other hand, is certain. Greedy decoding amplifies a single near-tie flip into completely different downstream text. You have to know this when running evals, regression tests or A/B comparisons, or you will mistake noise for signal.

`VLLM_SM86_DET_TOPK=1` provides a deterministic alternative selector (stable sort, ties resolved to the lower index), but it is very slow and is not recommended in production. It is also not "the correct answer": lower-index-wins is just another equally arbitrary tie break, and it carries a systematic bias toward the front of the sequence. Its legitimate use is pinning this variable down so that an A/B comparison means something, and that is exactly how this project used it to verify prefix caching correctness.

### A cache hit is not bit identical to a cold start

Even with the tie race above turned off, a cache hit on the same prompt is still not exactly the same as a full recompute: the token sequences begin to diverge after the first few tokens, and logprobs differ slightly at almost every position (on the order of 0.02-0.2).

We eliminated the suspects one at a time: turned off the upstream top-k tie race (residual unchanged), fixed a KV block zeroing defect that wrote out of bounds (unchanged), fixed an allocator asymmetry in delta-gather (unchanged), turned off the delta-gather path entirely (unchanged). The only explanation left is that this is simply the inherent numerical difference between "reusing a cached prefix" and "building the prefix block by block from scratch" — the tiling and reduction configurations of the attention and compression kernels differ between the two cases, so the floating point results differ. This is not specific to this fork; it is in the nature of prefix caching.

We measured how far the effect reaches: it **does not change retrieval correctness** (the cold and warm 40-cell needle grids are identical cell for cell, including the two cells that fail), and it **does not change the functional correctness of the cache** (hit length, zero deviation across 40 measurements). All it affects is whether the same input is guaranteed to give the same output, and on this stack that already does not hold because of the tie race in the previous section.

The reason we bring it up is that it cost us a considerable amount of time to chase, and chasing it turned up three genuine defects on the way (next section). If you see the same phenomenon in your own environment, there is no need to chase it a second time.

### Concurrency: at 262144 it is single stream only, and that is structural

With the cache on, the 262144 configuration can only serve a single stream. The reason is not "not enough memory" but the way the admission check reserves.

The measured behavior is this: **concurrency is real as long as each request stays within 10-20k**, and beyond that it degenerates into serialization. The boundary sits between 10752 (two requests interleaved, aggregate decode 66 tok/s) and 20992 (serialized, 23 tok/s).

The mechanism is that when the scheduler admits a new request it first has to reserve blocks for the sliding window group, and **that reservation grows with request length**. At short lengths both requests get in, their prefills interleave in the same step and finish together, and then they decode together; at long lengths the second request has to wait for the first to finish prefilling before it can enter, so when the first starts decoding the second is still prefilling, and a scheduling step filled by prefill advances decode by exactly one token — measured at 4.8 tok/s, against 50 for a single stream.

The most valuable observation from the diagnosis: when the second request is rejected, `vllm:kv_cache_usage_perc` reads only 0.49. **Usage counts blocks that have been written, the admission check counts blocks that have been reserved**, and they are not the same thing, so looking at usage makes it appear there is half the space free. By the same logic, we believed for a while that enlarging the pool would fix it, but at override 1600 NCCL failed to allocate buffers on the first use of the communicator (`ncclUnhandledCudaError` showing up in the PP broadcast rather than a clean torch OOM, so grepping for OutOfMemoryError returns nothing). 1200 is known safe, the ceiling is somewhere between the two, and that is not enough to admit two long requests at the same time.

PROFILE-P8 without the cache can do concurrency 2, because the ring placement compresses the compressor-state reservation into a constant (about 3 blocks per request) rather than something that grows with length. So this is a structural trade between the ring and concurrency.

Three related facts, recorded here so nobody falls into them:

**Aggregate prefill never exceeds single stream.** There is one token budget per step (F), and N concurrent requests share it rather than each getting one of their own. So what concurrency buys is fairness in latency, not more output from the machine. PROFILE-P8's concurrency-2 aggregate prefill of 4,909 against a single-stream 5,213 is exactly that relationship.

**`--long-prefill-token-threshold` has to be less than or equal to half of F**, otherwise a single prefill eats the whole budget and the second one never even gets the chance to queue in the same step. PROFILE-P8's threshold of 512 is exactly half of F=1024, and that is one of the necessary conditions for its healthy concurrency. But it is only necessary, not sufficient: dropping the threshold to F/2 (768→384) only recovered the 10752 point (decode 35→66, prefill 3035→4157) and changed nothing at longer lengths, because what is stuck there is the reservation, not the budget. Lowering the threshold is free for single stream — `VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1` only applies the cap when there are two or more prefills queued, and measured single-stream prefill is 4,646 vs 4,631, a difference within the noise.

**Concurrency 4 is not real concurrency at any length.** The early PROFILE-P8 concurrency-4 numbers (aggregate decode 24.7-43.2, clearly below concurrency 2's 63.5-80) were treated at the time as normal concurrency falloff; looking back, it is the same admission bottleneck, we just did not chase the root cause then.

Getting the ring's concurrency and absolute placement's caching at the same time would need a "boundary lead-in group" design: keep the ring, and hang an additional very small group off it that the prefix cache can address, holding only the 4 rows the compressor has to look back at before each alignment boundary. We evaluated it as feasible (roughly 7-9 calendar weeks, with the main risk being that it needs a mechanism for a group to abstain from the hit length, which has no precedent in the hybrid coordinator), but it is not in this release.

### Other limits

Concurrency 4 is not real concurrency at any length: the worst-case reservation for four requests is far beyond the pool ceiling. Our early PROFILE-P8 concurrency-4 numbers (aggregate decode 24.7-43.2, clearly below concurrency 2's 63.5-80) were treated at the time as normal concurrency falloff; looking back, it is the same admission bottleneck, we just did not chase the root cause then.

The FULL_AND_PIECEWISE cudagraph OOMs at 256K, short by about 2MiB. It fits at 131k, but measures no gain — its home ground is mixed steps (a step with one prefill and one decode), and a benchmark that starts its streams together cannot measure that.

Speculative decoding does not fit; the draft model wants another 0.9GB per card, and there is no room.

## Three defects found while chasing the residual

None of these were introduced by prefix caching. They were all pre-existing, and turning the cache on merely gave us the chance to observe them.

**KV block zeroing treats "block stride" as "write length."** DeepseekV4's packed layout has every layer of every group share one slab, with each layer being a strided view carrying its own byte offset. The zeroing kernel used `stride(block_dim)` as both the stride and the write length, so any layer with an offset greater than 0 writes offset bytes too many into the next block on every zeroing — and that block may be in use by another request — while the last block writes past the end of the allocation entirely. The fix splits stride and payload into two tables, with payload computed from the stride span (so that a K/V-first reordered layout also stays inside the same block), plus an assertion at init so that an illegal geometry fails loudly instead of corrupting silently.

**The zeroing predicate is an exact type() comparison and misses the sliding window family.** The SWA KV window and the fp32 compressor state are not in that tuple, so their new blocks were only zeroed when they happened to be aliased by some other group's tensor. Changed to `isinstance(AttentionSpec)`, consistently on both sides (scheduler and worker).

**delta-gather's `empty_cache` has no cap on how often it fires.** It originally keyed off "the edge of the prefill set", so a request that was skipped for a step and then came back would flush the allocator again, and `empty_cache` is a device-synchronizing call. With three or more concurrent prefills this happens over and over. Changed to a once-per-request latch. In the same file, the freshness test for the BLOCKED sentinel was itself defeated by prefix caching (a cache hit makes the first sighting arrive with a nonzero prefix); it now uses the same equality contract as the tracking path.

All three have regression tests, and each test was only accepted after it had been verified to fail against the pre-fix code.

## Findings against upstream

Two problems in upstream vLLM turned up during development. Both are fixed in this fork.

**Silent pipeline-parallel desync.** Exceptions on non-output ranks were swallowed, which caused the isend to be skipped and produced a permanent +1 offset. It does not crash, it just quietly emits wrong content, which is the hardest class of bug to track down. The fix rethrows the exception and adds a PP step-id contract to the tensor dict as a hard check.

**Worker exceptions swallowed.** `multiproc_executor` handles exceptions on non-reporting ranks silently, so an error such as running out of memory presents as "the engine is hung" rather than a clear failure, and recovery requires a manual pkill.

## License and Credits

This fork inherits vLLM's Apache 2.0 license. New files carry `SPDX-License-Identifier: Apache-2.0`.

The flash-mla patch branch is MIT (Copyright (c) 2025 DeepSeek), compatible with Apache 2.0.

- Built on [haosdent/vllm](https://github.com/haosdent/vllm) — DeepSeek-V4-Flash support for vLLM
- The compression-preserving DCP design derives from Lasimeri's context-parallelism work; attribution notes are in `sm86_dcp_layout.py`, `sparse_attn_indexer.py` and `dcp.py`
- The flash-mla sm86 sparse kernel integration is built on [AppMana/forks-flash-mla-int](https://github.com/AppMana/forks-flash-mla-int), the flash-mla fork this branch derives from. The four P9 commits this fork adds on top live at [tomylin890/flash-mla-sm86-dsv4](https://github.com/tomylin890/flash-mla-sm86-dsv4), branch `dcp-sm86-patches` — that is the one to build
- [AppMana/forks-vllm-consumer-nvidia-platforms](https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms) was consulted as a reference for what is achievable on consumer GPUs. No code was taken from it, and the tree contains no reference to it
