## What DeepSeek-V4-Flash's KV actually looks like

To understand what this branch does you first have to know how far this model's KV is from ordinary attention. Every item below follows the code (`vllm/models/deepseek_v4/`); where the paper and the code disagree the code wins, and the conflict list is in section 9 of `ARCHITECTURE.md`.

### Called MLA, actually shared-KV MQA

`num_key_value_heads=1`, and one 512-dim KV entry serves as both K and V; the config has no `kv_lora_rank` and no `v_head_dim`. vLLM marks it `is_mla()=True` only so that the MLA cache plumbing can be reused. Do not carry the V3 MLA assumptions over.

head_dim 512 = 448 dims of NoPE + 64 dims of RoPE, there are 64 query heads, and the softmax scale is 1/√512. An entry occupies 584 bytes in the cache: 448 dims in FP8 (quant block 64, so 7 real UE8M0 scales plus 1 pad byte) plus 64 dims of BF16 RoPE, with the data segment aligned to 576. `compressor.py`'s `_token_stride` (448+64×2=576) and `_scale_dim` (448//64+1=8) are exactly this arithmetic.

### Two compression ratios, two completely different properties

43 layers in the backbone. L0 and L1 are pure sliding window (`compress_ratio` is 0, and the code takes `max(1, ...)` and treats it as 1); L2 through L42 alternate between CSA (m=4) and HCA (m'=128), 21 C4 layers and 20 C128 layers. Another 3 DSpark draft layers are pure sliding window.

C4 is **overlapping** compression: each entry is a weighted sum of 2m=8 tokens, reaching back into the previous block, with the weights coming from a single joint per-dimension softmax (each of the 512 dimensions gets its own convex combination). C128 does not overlap. In the code this is `self.overlap = compress_ratio == 4` and `self.coff = 1 + self.overlap`, and `coff` runs from there through every dimension and window computation.

The remaining rules all become constraints once you shard:

- A block completes when `(pos + 1) % m == 0`, and an entry's RoPE position is `m·i`, the absolute position of the block's first token.
- The position bias `ape` is indexed by **absolute position mod m**. Any resharding must preserve position-mod-4 alignment.
- Causal visibility is `s < (t+1)//m`: a query can see the block it has just completed. The paper says `Floor(t/m)`; the code does not.

### The sparse indexer

Only C4 layers have an indexer (`attention.py`: `if self.compress_ratio == 4`). It has a complete compressor of its own: head_dim 128, the same m=4, its own W and `ape`, quantization after a Hadamard rotation. The cache is FP8 at 132 bytes/entry (128 bytes plus a 4-byte fp32 scale); there is also a 68-byte MXFP4 layout, but SM8x is not allowed to use it and the DCP write path raises `NotImplementedError` on it directly.

The score is `Σ_h w_h · ReLU(q_h · k)`, with **no softmax**, over 64 heads. top-k is global, and k comes from `config.index_topk` (512 for Flash). SM8x goes through the Triton fallback (`fp8_mqa_logits_triton` / `fp8_paged_mqa_logits_triton`, fp32 logits).

"No softmax" plus "ReLU" is where the tie race in the README's "Known behaviors" comes from: an entry that matches no head scores exactly 0.0.

### The sliding-window groups

Window 128, block 64. The block size is not a free parameter: SWA and C4A KV share the same physical tensor, and C4A's block shape is `[256//4, head_dim] = [64, head_dim]`, so SWA has to be 64 as well.

It **shares one softmax with the compressed branch**: the indices are concatenated and there is a single sparse attention call. Under DCP this means SWA's contribution has to enter the merge as one (m, l, o) partial, and must never be normalized on its own.

The attention sink is one learnable fp32 logit per head, **added to the softmax denominator only, only once, and against the global running max**.

### fp32 compressor state

The compression computation is fp32 all the way through: W^KV, W^Z, `ape`, the state buffer, the softmax, the weighted sum, and only then RMSNorm → RoPE (last 64 dims) → FP8 quantized write. Doing the compression in bf16 biases every single entry.

The key point is that this state is **itself a paged KV group**, with spec `SlidingWindowMLASpec` and a hard dtype assertion on fp32 (`CompressorStateCache`). Its geometry is bound by page sharing too:

| State group | state_dim | block_size | sliding_window | Per token |
|---|---|---|---|---|
| C4 compressor (attention) | 2·coff·512 = 2048 float | 4 | 8 | 8192 B |
| C128 compressor | 2·1·512 = 1024 float | 8 | 128 | 4096 B |
| C4 compressor (indexer) | 2·coff·128 = 512 float | 4 | 8 | 2048 B |

At every boundary position `p` the compression kernel gathers the `L = sliding_window` rows `[p − (1 + OVERLAP)·m + 1, p]`, of which `L − m` rows fall below `p`. This one fact drives both P8 and P11, so it is worth committing to memory now.

### The groups added up

| Group | Spec | Block | Contents | DCP |
|---|---|---|---|---|
| C4A compressed KV | `MLAAttentionSpec` cr=4 | 256 token / 64 entry | 584 B/entry uint8 | sharded |
| C128A compressed KV | `MLAAttentionSpec` cr=128 | 256 token / 2 entry | same | sharded |
| Indexer KV | `MLAAttentionSpec` cr=4 | 256 | 132 B/entry | sharded |
| SWA KV | `SlidingWindowMLASpec` window 128 | 64 token | raw KV, shares physical pages with C4A | replicated |
| C4 compressor state | `SlidingWindowMLASpec` fp32 window 8 | 4 token | 8192 B/token (2048 for the indexer version) | replicated |
| C128 compressor state | `SlidingWindowMLASpec` fp32 window 128 | 8 token | 4096 B/token | replicated |

The scheduler side merges groups with identical geometry, so the real scheduler group count is lower than the table suggests (the indexer's C4 state has the same geometry as the attention C4 state). What you actually have to remember is that set of block sizes: **256, 64, 4, 8**. It determines `scheduler_block_size = lcm(each group's block × dcp)`, which is 1024 at dcp=4 (this is the "hit length aligned to 1024" in the README) and 2048 at dcp=8, and it also determines that once caching is on, `hash_block_size = gcd(...)` can only be 4.

## Why vLLM's context parallelism does not connect

Upstream DCP assumes one homogeneous full-attention KV group sharded along the sequence dimension. What is here is a mixed set of six geometries and three dtypes, of which only three groups **can** be sharded. It gets stuck in four concrete places.

**1. Group type.** `HybridKVCacheCoordinator` asserts that every group is a `FullAttentionSpec` or a `MambaSpec` when `dcp_world_size > 1`. `MLAAttentionSpec` inherits from `FullAttentionSpec`, so the three compressed KV groups get through; `SlidingWindowSpec` inherits directly from `AttentionSpec`, so SWA and both state groups hit the wall. This is a blanket guard for "no DCP-aware handling yet", not something aimed at this model.

**2. Compressor state cannot be sharded.** It is not KV. It is an fp32 accumulator addressed by absolute position that looks back `L` rows at every boundary, and round-robin sharding chops the lookback window into pieces. SWA is the same story: every rank has to see all 128 tokens, because it shares a softmax with the compressed branch. So a notion of "this group is replicated, not sharded" is needed, and that is what `dcp_exempt` is.

**3. The SM8x decode kernel does not emit LSE.** `rocm_sparse_attn_decode` treats SWA as the main segment and the compressed top-k as the extra segment, runs a single online softmax, applies the sink inside the kernel and writes only normalized bf16; `m_i` / `l_i` stay in registers and are thrown away. Without (m, l) there is no way to merge across ranks.

**4. The indexer's top-k is global.** Each rank taking the top-512 of its own shard does not union to the global top-512. Upstream refuses this in writing: `vllm/v1/attention/backends/mla/indexer.py`, `NotImplementedError: DCP is not supported with sparse indexer KV compression`.

Layer the precision constraints on top of that and these stop being "can it be connected" problems and become "is it right once connected" problems: the sink may only be applied at the global merge, against the global max, once; the merge must be fp32 with a fixed rank order (DeepSeek explicitly disables split-KV attention for batch invariance, so determinism is ours to preserve); global absolute position must run through `ape`'s mod-4, the entry's RoPE position and the YaRN table, and renumbering from 0 per shard is a silent precision bug; the KV bytes are part of the function, so quantize once, move the data in quantized form, and never requantize with a different grouping (this layer's quant block is 64, DeepGEMM's default is 128). They are written up as 12 rules in section 10 of `ARCHITECTURE.md`; use them as a checklist while implementing.

## The shape of the sharding

Sharding happens in **compressed entry space**, not token space. The single source is `vllm/v1/attention/backends/mla/sm86_dcp_layout.py`:

```
owner(e)       = (e // I) % W
local_entry(e) = (e // (I·W))·I + e % I
global(r, j)   = (j // I)·(I·W) + r·I + j % I
```

`W` is the dcp world size and `I` is `cp_kv_cache_interleave_size`, locked to 1 throughout (upstream warns that interleave > 1 fails gsm8k parity even on V3.2; do not touch it). `-1` is the invalid sentinel and passes through every helper unchanged.

Order matters: **divide by m first, then localize.** The other way around undercounts. With world=2, rank 0, m=4 and a total length of 12, the right answer is 2 entries ({0, 2}), but `localize(12)//4` gives only 1. P2b fixed this.

Shard boundaries have to land on multiples of 128, which `scheduler_block_size` (1024 at dcp=4) satisfies automatically.

## What each phase did

The order here is the actual development order; each step began only after the previous step's gate passed. Sections P1 through P9 each correspond to a `P*-NOTES.md`, P11 and P12 correspond to `-DESIGN.md` files, and P10 exists only in commit messages. Two numbers are skipped in the middle: P3 was the capacity validation gate in `PLAN.md` and left behind only a fix commit reconnecting the short-context top-k fast path to the DCP producer, with no document of its own; P5 evaluated the Marlin INT8 activation path for the MoE and **rejected** it, with the conclusion recorded in `P5-NOTES.md` and zero code changes.

### P1 — KV group plumbing and dcp_exempt

Problem: the engine would not start.

The fix is a single predicate: `is_dcp_exempt_spec` (`vllm/v1/core/kv_cache_utils.py`). Every place that has to agree on "is this group sharded or replicated" calls it — the manager's real block size, the scheduler-side group block size, the worker-side block table row width — so they cannot drift apart. The surrounding plumbing:

- `vllm/v1/kv_cache_interface.py`: the `dcp == 1` assertion in `SlidingWindowSpec.max_memory_usage_bytes` is allowed through under the gate (the unsharded per-rank size is exactly the replicated footprint); a new gated `max_num_blocks_per_req` returns the unsharded row width.
- `vllm/v1/worker/block_table.py`: `BlockTable` gains a `shard_dcp` parameter (forcing world=1 / rank=0, so `compute_slot_mapping` stores every token locally), and `MultiGroupBlockTable` gains a `dcp_exempt` list.
- Threaded through `gpu_input_batch.py` to `gpu_model_runner.may_reinitialize_input_batch`, which computes exempt per group and folds it into the comparison that triggers reinit.

What this bought: dcp=2 initializes, and the KV pool shows roughly 2× the token count with KB/token still at the compressed level. The latter is the check for "silently lost the compression", and it matters more than the former.

The same step planted a landmine: `resolve_kv_cache_block_sizes` multiplies every `AttentionSpec` by dcp, including the exempt groups the manager skips scaling for. With caching off it is completely invisible; with caching on it explodes at startup. See B2 under P11.

### P2a — cross-shard merge on the decode side

Problem: the kernel does not emit LSE.

`rocm_sparse_attn_decode` gains a `return_softmax_stats=True` (an `EMIT_SOFTMAX_STATS` constexpr inside the kernel) and emits pre-sink fp32 `(m, l)` of shape `[T, H]`. A new file, `vllm/models/deepseek_v4/common/ops/dcp.py`, turns that into `lse = m + log(l)`, hands it to the in-tree `cp_lse_ag_out_rs` / `dcp_a2a_lse_reduce`, and finally `apply_attn_sink` uses `logaddexp` to fold the sink in once against the global max.

Two details that are easy to get wrong: the sentinel for an empty shard is a finite `-1.0e30` rather than `-inf` (`-inf − -inf = NaN` poisons the entire softmax algebra, whereas `exp(-1e30 − g)` is simply 0 in fp32, so what should drop out drops out naturally); and the LSE base convention is natural log throughout (`is_lse_base_on_e=True`), where getting it wrong destroys the result silently instead of raising.

Q has to be all-gathered (`_maybe_gather_dcp_q`): TP splits the query heads and DCP splits KV inside the TP group, so each rank's KV shard has to face all of the group's heads; the a2a LSE merge then sends the head outputs back to their owner rank.

What this bought is a decode numerically equivalent to "a single global softmax with the sink applied once". It was checked against a pure-python simulation on four cases (including two empty shards, all-empty with sink, and sink close to max), with a maximum absolute difference of 2e-14.

### P2b — global top-k

`_sm86_dcp_global_topk` (`vllm/model_executor/layers/sparse_attn_indexer.py`): each rank takes a local top-512 of its own shard, all-gathers fp32 scores plus int32 global entry ids in a fixed rank order (fixed width `topk × dcp`), then takes the global top-512.

`torch.topk` does not guarantee tie order, so it was replaced with two stable argsorts: ascending on the global index first, then a stable descending sort on score, so equal scores take the smaller global entry index. Invalid candidates get score `-inf` and the maximum `int32` sort index, so that a candidate which genuinely scored `-inf` still beats padding.

The acceptance criterion is "the selected index sets are exactly equal" plus token-identical greedy decoding, **not** logit MSE. Floating-point differences in the scores are meaningless here; which entries get selected is not.

The same phase also fixed the compress-then-localize order for decode seq len, added a rank-local compressed entry slot mapping (`_sm86_dcp_compressed_slot_mapping`), and changed the indexer's `compress_ratio > 1` DCP guard to pass under the gate.

### P2c — the write side

Writing compressed entries has to be filtered by ownership with the slot rewritten. Under the gate the compressor's forward is forced onto the Triton launcher (the cutedsl SM90+ path and the ROCm two-stage path have no CP layout support); the kernel receives three constexprs, `DCP_WORLD_SIZE` / `DCP_RANK` / `DCP_ENTRY_INTERLEAVE`, computes the global entry index `e = position // m` for each boundary token, returns early if it does not own it, and if it does own it, computes the rank-local slot inside the kernel from P1's sharded block table.

Every rank computes **every** entry in fp32 and stores only the ones it owns. The computation is redundant, but it needs no collective at all and is therefore capture-safe. That trade is deliberate.

### P2d — prefill all-gather and C128A decode metadata

This step lifted the ownership algebra out of the indexer and made `sm86_dcp_layout.py` the single source (a pure move, no logic change). The prefill gather, the C128A decode metadata, P4's virtual block table and P7's delta planner all take the same formulas from there afterward.

### P2e — deterministic top-k (for debugging)

`VLLM_SM86_DET_TOPK`, with the selector in `vllm/v1/attention/ops/sm86_det_topk.py`. Why it exists, and why it is not "the right answer", is already covered in the README's "Known behaviors". One thing to add here about its use: when verifying prefix caching correctness, it is what pins the upstream tie race down so that an A/B comparison means anything.

### P2f — CUDA graph

P2a had put an unconditional `RuntimeError` under `torch.cuda.is_current_stream_capturing()`. P2f took it apart: classify op by op (in-place, persistent buffer, graph-private pool, host constants) and route every output into a persistent buffer allocated outside capture.

Along the way this dug out the real reason: the C128A decode width in `sparse_mla.py` was not pinned. That is the thing that could not be captured. P2a's stated reason ("every step is a new ragged tensor") does not actually hold on this base, because non-DCP C4A decode was already running inside capture.

What this bought is DCP decode running a `FULL_DECODE_ONLY` graph: 4-6 tok/s eager, around 70 with graphs on.

### P4 — the fixed overhead in prefill

Diagnosis before fix. DCP prefill is about 3.3× slower than dcp=1, and the per-forward cost is **flat** across a 5.6× range of prefix lengths (360-460 ms). Cost that does not vary with length is the signature of a fixed per-layer overhead: not bandwidth, and not a slow kernel. One 2730-token prefill at F=256 measured roughly 72,800 extra kernel launches, 7,216 hard syncs and 913 NCCL collectives.

Four optimizations:

1. The virtual block table became closed-form instead of being stacked up rank by rank with 16 `nonzero()` calls (those 16 calls were the 16 hard syncs).
2. Memoize it. It depends on six values only, `(max_entries, max_local, num_reqs, world, interleave, device)`, is independent of the layer and of the KV data, and yet was being rebuilt once per forward for each of the 41 compressed layers. The LRU is bounded on both entry count and element count, because chunked prefill drives `max_entries` steadily upward and the keys really do churn.
3. The indexer DCP merge short-circuits when the total number of global compressed entries does not exceed topk: every candidate will be selected, the merge is the identity, and the whole thing can be skipped, two NCCL calls included. The quantity the gate uses is the host-side `max_seq_len // compress_ratio`, defaulting to `-1` for "not provided", so when it is not supplied the short-circuit is off rather than being misread as 0.
4. `get_dcp_local_seq_lens` no longer wraps `dcp_rank` in a device tensor. `torch.tensor(scalar, device=cuda)` does one blocking H2D from pageable memory, and this path runs once per compressed layer per chunk.

The result is that a cache-hit layer (39-40 of the 41) does 0 torch ops, 0 syncs and 0 H2D on this path.

### P6 and P9 — the flash-mla sm86 kernels

P6 is prefill: a single `flash_mla.sparse_mla_prefill` (`torch.ops.flash_mla.fwd_sparse_prefill_mla`) replaces the original three-piece "bf16 dequant workspace + `combine_topk_swa_indices` + `rocm_sparse_attn_prefill`", once per chunk per layer. The SWA and compressed streams share a single softmax inside the kernel (the hard rule in `ARCHITECTURE.md` section 6), and the sink is folded in once at softmax initialization. The adapter is `vllm/models/deepseek_v4/ampere/flash_mla_prefill.py`, behind the flag `VLLM_DSV4_FLASH_PREFILL`.

P9 is decode: a **new** op, `fwd_sparse_decode_mla_partial`, added to the flash-mla fork, emitting per-rank normalized pre-sink `out` plus natural-log fp32 `lse`, which is exactly the shape P2a's merge path wants. So on the vLLM side this is only swapping `rocm_sparse_attn_decode + softmax_stats_to_lse` for a single op call; `dcp_merge_flashmla_output` and `self.attn_sink` were not touched at all. The existing ops were left alone. The adapter is `ampere/flash_mla_decode.py`, behind `VLLM_DSV4_FLASH_DECODE`. The same batch of patches also added a `BLOCK_M=16` prefill instantiation, dispatched when `num_heads <= 16`.

The README's "decode barely moves from short context to 254k" comes from here: the cost depends only on the 512 selected entries, not on total sequence length.

P6 came with two side pieces: `VLLM_DSV4_WARMUP` (on by default) runs the whole resolved family of prefill/decode kernels once at engine init, with the catalog in `vllm/model_executor/warmup/dsv4_sm86_warmup.py`; and `VLLM_DSV4_SM86_INDEXER_TILES` exists because the upstream Triton indexer autotune configs were tuned for A100/SM80. P9 additionally pinned down the cause of prefill recompiling once per chunk-count bucket (a constexpr that varied with the bucket). These address different layers of the same thing, and the deployment warmup SOP described in the README's "Memory traps on 24GB cards" is the residue that remains after all of them.

The flash-mla side of the changes lives on a separate fork branch (`dcp-sm86-patches`), built into the venv with `FLASH_MLA_CUDA_ARCHS=86`; the build steps are in install.md, which also records the remote that branch is cloned from.

### P7 — delta gather

Problem: `ampere_sparse.py` passed `gather_lens=None` into the compressed entry gather, so every prefill chunk re-packed, re-all-gathered and re-consumed the **entire** prefix. A compressed entry never changes once it has been written at its block boundary, so everything after the first pass is redundant retransmission, `O(P²/(m·F))` in total.

The approach: one persistent `[capacity, 584]` uint8 staging buffer per (request, layer), where row `e` holds the 584 raw bytes of global entry `e` (576 of data plus the UE8M0 scale at +576, which is exactly one page at `block_size = 1`). Row numbers are stable across chunks, so the index conversion degenerates to the identity. That is the reason for choosing global entry order rather than per-chunk gathered order: with the latter, every id shifts as soon as `max_local` grows. Each chunk packs, all-gathers and scatters only the newly added `[prev, new)`. Bytes and scales are moved verbatim throughout, and the only dequant is still the original one.

The lifecycle does not rely on a scheduler hook (`vllm/models/deepseek_v4/ampere/dcp_delta_tracker.py`): the GC condition is "did not appear in this step's prefill rows", and finishing, aborting, preemption and moving into decode all present as absence. On top of that, a prefix continuity check makes stale state structurally impossible: the prefix token count at each sighting must equal the seq len recorded at the previous sighting, chunked prefill guarantees that, and any violation resets to a fresh admission. There is a per-worker byte budget (`VLLM_DSV4_DELTA_GATHER_BUDGET_MB`); a failed charge sends that (request, layer) back to a full re-gather, which is per-request per-layer graceful degradation rather than overall failure.

The third of the three defects in the README (unbounded number of `empty_cache` triggers) is in this file.

### P8 — a ring buffer for compressor state

Problem: the default layout gives every scheduled token a paged row, so a step with F tokens forces `F + L − 1` live rows, and the per-request reservation grows linearly with F.

The approach: a ring of W tokens with `slot = position % W`, and each step of the compressor forward cut into sub-chunks of `G = W − L + 1` flat batch tokens. `L` is the kernel's gather span (`coff · m`). Within a sub-chunk every write happens before every read, and the live set is at most `(G − 1)` new writes plus `(L − 1)` lookback plus 1, that is `G + L − 1` distinct positions, so `W ≥ G + L − 1` is collision-free. The bound is **tight**: a sub-chunk starting exactly at a boundary position starts breaking at `G = W − L + 2`.

Cutting on the flat token axis is the key to per-request safety: each request's tokens are a contiguous increasing run inside the batch, so a flat slice of G tokens contributes at most G **contiguous** positions to any one request.

The reservation therefore collapses to the constant `cdiv(W, block_size)`, which is the 858 → 262 in the README.

All of the validation is a pure function of `vllm_config`, so every rank computes the same answer and the admission numbers the scheduler sees stay rank-invariant. Both rejections are hard raises rather than degradations:

- Mutually exclusive with prefix caching (the README already gives the reason: a `pos % W` slot is not prefix-addressable in the first place). Degrading to `None` would change the KV footprint by several GB, and would surface only much later as an admission rejection or an OOM.
- The hybrid KV cache manager is required to be on. `_promote_local_kv_cache_specs` rebuilds `SlidingWindowMLASpec` into `MLAAttentionSpec` and cannot carry `state_window` across, so the write side (slot mapping, derived from the spec) goes back to absolute positions while the read side (the compressor, derived from the env) keeps folding: a silent numerical error. That flag can also be set **implicitly** (a KV connector without HMA support, a platform without hybrid support), not only from the CLI.

### P10 — adaptive long-prefill threshold

`--long-prefill-token-threshold` takes effect globally. On a single stream it doubles the chunk count, taking the 200k needle from 44 seconds to 59 seconds (-34%). But it is at the same time a necessary condition for concurrency health (from the README's concurrency section: the threshold has to be less than or equal to half of F, or a single prefill eats the entire budget).

`VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE` (off by default, matching upstream semantics) applies the cap only when two or more requests in the prefill stage are queued; a single request keeps the full F. The pending count is the waiting queue plus running requests with `num_computed < num_prompt`, computed once per `schedule()`. The change is in `vllm/v1/core/sched/scheduler.py`.

### P11 — prefix caching

The research in this phase overturned the original design, and it is worth recording in full, because both negative results are counterintuitive.

The original plan was "after a hit, trim back a few blocks and recompute".

**Trimming is not enough under the ring.** The compression kernel's gather is `start = position − (1 + OVERLAP)·m + 1`, and the only guard is `mask_pos = pos >= 0`. With m=4 and L=8 for C4, the first boundary after any m-aligned resume point R is `p = R + 3`, and its gather starts at `R − 4`. Those four rows `[R−4, R−1]` are read and never recomputed. Trim 256, 1024 or 8192 and the gap merely moves somewhere else; the deficit is scale-invariant. On top of that, newly allocated state blocks were not on the zeroing list at the time (`SlidingWindowMLASpec` was deliberately excluded from that `type()` comparison), so those four rows were uint8 KV bytes read as fp32, which puts the severity at Inf/NaN rather than "slightly off".

**Trimming is unnecessary under the default layout.** `SlidingWindowManager._contiguous_blocks_for_hit = cdiv(window − 1, block_size)` preserves exactly the lookback window at a hit boundary H: for C4 state, `cdiv(7,4)=2` blocks, that is `[H−8, H)`, which strictly contains the required `[H−4, H−1]`; for C128 state, `cdiv(127,8)=16` blocks, that is `[H−128, H)`, and C128 has no overlap, so its first boundary `p = H+127` reads `[H, H+127]` and needs nothing below H; for SWA KV, `cdiv(127,64)=2` blocks is the entire 128 window. And H is a multiple of `scheduler_block_size`.

So prefix caching was not obtained by trimming; it was obtained by **giving up the ring layout**. That is where the mutual exclusion in the README's section on choosing between the two profiles comes from.

The actual blockers were two startup errors with nothing to do with the compressor:

- **B1 (upstream)**: the `HybridKVCacheCoordinator` type assertion described earlier. The only reason it is invisible today is that with caching off, `get_kv_cache_coordinator` routes to `KVCacheCoordinatorNoPrefixCache`.
- **B2 (ours, created in P1)**: `resolve_kv_cache_block_sizes` multiplies every `AttentionSpec` by dcp, including the exempt groups. With caching on the function no longer early-returns, so `hash_block_size = gcd(1024, 256, 16, 32) = 16` while the manager's block sizes are `[1024, 64, 4, 8]`, and the coordinator's divisibility assertion blows up on `4 % 16`. The comment P1 left behind said "the coarser LCM is a safe superset alignment, no gate needed here" — true for LCM, exactly the wrong direction for GCD. In practice B2 fired before B1, because the divisibility assertion comes before the type assertion.
- **B3 (a consequence, not a bug)**: after the fix, `hash_block_size = gcd(1024, 64, 4, 8) = 4`, pinned by the C4 state group's page-sharing-constrained block_size of 4, with no coarser option available. A 262k prompt therefore has to compute about 65,536 chained block hashes on the scheduler's critical path.

The config-time checks live in `compressor.py`: `validate_compressor_lookback_coverage` turns the arithmetic above into assertions that fail loudly (the hit reservation must cover the lookback, and `scheduler_block_size` must divide every window and every m), and `check_compressor_kv_cache_config` prints the profile name and the three block sizes into the startup log, so the deployment matrix can be asserted straight from the log. That these conditions hold today is an arithmetic coincidence of 1024/128/8/4; if any one of those numbers moves it should blow up, and that is exactly why both functions exist.

### Not done: P12

The "boundary lead-in group" mentioned in the README: keep the ring, and hang beside it a very small, prefix-cache-addressable group that stores only the few rows the compressor has to look back at before each aligned boundary. The full design (11 implementation steps, adversarial review, effort and risk) is in `P12-DESIGN.md`. If you are picking up "have the ring and the cache at the same time", start there rather than from scratch.

## Where to start reading

The branch is `dcp-sm86` and the base is `f8ea5bb16` (haosdent's DSV4 SM8x support). `git diff f8ea5bb16 HEAD` is the whole branch: 43 files, all `.py`, of which 11 are new (7 source, 4 tests; 1 existing test was also modified).

| File | What it handles |
|---|---|
| `vllm/v1/attention/backends/mla/sm86_dcp_layout.py` | the single source of the ownership algebra, read this first |
| `vllm/models/deepseek_v4/common/ops/dcp.py` | cross-rank LSE merge and sink |
| `vllm/models/deepseek_v4/ampere/ampere_sparse.py` | the main entry point for the prefill/decode DCP paths |
| `vllm/models/deepseek_v4/compressor.py` | compressor, state group spec, P8's ring, P11's config-time checks |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | the fused compress/quantize/write kernel, the lookback window is here |
| `vllm/models/deepseek_v4/ampere/dcp_delta_tracker.py` | P7's staging lifecycle |
| `vllm/models/deepseek_v4/ampere/flash_mla_{prefill,decode}.py` | adapters for the flash-mla ops |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | the indexer's global top-k merge |
| `vllm/v1/attention/backends/mla/indexer.py` | DCP localization of indexer metadata |
| `vllm/v1/core/kv_cache_utils.py` | `is_dcp_exempt_spec`, block size resolution |
| `vllm/v1/worker/block_table.py` | `shard_dcp` / `dcp_exempt` / the ring's slot mapping |

Document mapping: `PLAN.md` (phase split and gates), `ARCHITECTURE.md` (model and precision hard rules), `PORT-NOTES.md` (P1 and subsequent fix records), `P2a` through `P2F` (decode-side DCP), `P4` (prefill overhead), `P5` (the rejected Marlin INT8 path), `P6` / `P9` (flash-mla), `P7` (delta gather), `P8` (the ring), `P11-DESIGN.md` (prefix caching research and design), `P12-DESIGN.md` (the unimplemented lead-in group).

Every notes file states explicitly which conclusions are static derivation and which were measured on the machine. This development box has no CUDA, so many phases were completed statically first and verified on hardware afterward; keep that distinction in mind while reading.
