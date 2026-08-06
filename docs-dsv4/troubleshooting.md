## Troubleshooting

Every entry below is something this project actually hit. Symptom first.

Before you start digging, pull these three things out of the boot log; almost every entry below uses them:

```
DeepseekV4 serving profile PROFILE-CACHE (compressor-state ring: off, prefix caching: on,
max_num_batched_tokens: 768), scheduler_block_size=1024, hash_block_size=4,
num_gpu_blocks=..., sliding_window_groups=...
Available KV cache memory: X GiB
GPU KV cache size: N tokens
```

The first line comes from the profile decision in `compressor.py` and tells you directly whether this boot came up as PROFILE-P8 or PROFILE-CACHE, so you don't have to guess whether an environment variable took effect; the numbers in the back half of it are what you assert the deployment matrix against (a block size problem of the B2 kind shows itself here first). The second line is the **physical** amount available. The third is the pool size after `--num-gpu-blocks-override` is applied.

### The engine stops responding, and the log repeatedly prints "No available shared memory broadcast block found in 60 seconds"

That line is an INFO printed by `shm_broadcast.py` once every `VLLM_RINGBUFFER_WARNING_INTERVAL` (60 by default) seconds while it spins waiting. The "60 seconds" in the message is that interval constant, not how long it has already waited — after ten minutes it still writes 60, so don't read this line as "only stuck for a minute."

What it means is that some peer isn't making progress. During startup, while kernels compile or weights are quantized, it is normal. In service it means one worker has already died or hung, and the other workers are still waiting on it.

Scroll up in the same log and look for these two lines:

```
WorkerProc hit an exception.
Worker proc <name> died unexpectedly (exit code: <n>), shutting down executor.
```

The first is the worker's own traceback, and that is the real cause (on this machine, most often an OOM partway through the prefill of a long request). The second is the parent process's monitor thread noticing it died. Upstream vLLM only let `output_rank` report exceptions; exceptions on non-reporting ranks were swallowed, so the only symptom was the engine quietly hanging. This branch re-raises on non-reporting ranks (see "Findings against upstream" in README.md), so the traceback now always gets printed, but you have to go up and find it yourself.

What to do: `pkill` (next entry), then restart. This state does not recover on its own.

Separately, ssh being unreachable at a moment like this is not an independent event. Eight hung workers are still spinning on NCCL, all 16 threads of the 5700X are saturated, and sshd gets sluggish or refuses connections entirely. **The ssh failure is a symptom, not a cause** — don't rush to conclude the machine is dead or go cut power; work from a session you already have open, or wait for the spinning processes to be killed.

### The engine doesn't clean up after itself when a worker dies

Even when the monitor thread prints `died unexpectedly ... shutting down executor`, teardown itself still hangs on the shm broadcast wait: the processes stay, and the VRAM is not released. This is a known gap and is not fixed yet.

So cleanup cannot rely on `Ctrl-C` alone:

```bash
pkill -9 -f "VLLM::"
pkill -9 -f "vllm[ ]serve"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

All three are necessary, each for a different reason:

The `VLLM::` one cannot be skipped. Worker process titles look like `VLLM::Worker_PP0_TP0_DCP0` (`set_process_title` prefixes them with `VLLM::`), and `pkill -f "vllm serve"` only reaches the launcher. The workers survive, each card keeps holding twenty-some GB, and the next boot OOMs in `set_device_index`.

The brackets in `"vllm[ ]serve"` are not a typo. `pkill -f` matches against the full command line, so if you run `ssh host 'pkill -f "vllm serve"'`, the remote `bash -c`'s own command line contains the string `vllm serve`, so it kills itself, ssh drops on the spot, and the real engine is still alive. Writing it as `vllm[ ]serve` unties that: the regex still matches the real `vllm serve` command line, but its own command line carries the literal text `vllm[ ]serve`, which the regex does not match, so the self-match is broken. For the same reason, **do not put the kill and the launch in one ssh command**.

The VRAM check is a completion condition, not a courtesy. When `pkill -9` returns, the processes have not necessarily released their memory yet; you have to poll until every card has dropped back under 500MiB before starting the next round, otherwise the next boot does its memory profiling against a polluted available figure. The sequence is: `pkill -9`, then poll `nvidia-smi --query-gpu=memory.used` until every card is under 500MiB, and only then boot. The script writes `vllm serve` without brackets, because it runs directly on the machine and its own command line doesn't contain that string; you only need the bracket form above when going in through `ssh host '...'`.

### ncclUnhandledCudaError at boot in the PP broadcast, but grep OutOfMemoryError finds nothing

This is `--num-gpu-blocks-override` set too high. Measured: an override of 1600 reproduces it every time, 1200 is known safe, the ceiling is somewhere between the two.

Mechanism: NCCL's communication buffers are allocated lazily, on first use of the communicator, and that moment falls after the KV pool has been allocated. The pool has eaten VRAM down to scraps, NCCL can't get its share, and it reports its own error code rather than torch's `OutOfMemoryError`. So this is an out-of-memory problem that shows up under none of the keywords you are used to grepping for.

The way to identify it is exactly this combination:

```bash
grep -c OutOfMemoryError <log>   # 0
grep -n ncclUnhandledCudaError <log>   # hits, with the surrounding context in the PP broadcast
```

When you see that combination, lower the override; do not raise `--gpu-memory-utilization` — util has no effect whatsoever on actual usage while the override is in force.

### Boot is refused, the message says available is down to 0.18 GiB, when there is clearly more than that

The error looks like this:

```
To serve at least one request with the model's max seq len (262144), (0.55 GiB KV cache is
needed, which is larger than the available KV cache memory (0.18 GiB).
```

The `available` here is **the effective capacity after the override is applied**, not the physical amount available. One instance I actually walked into: physical was 1.74 GiB, the override was set to 400, and the message read 0.18. It looks like not enough memory; in fact I had made the pool too small myself.

For the physical number, find this line:

```
Available KV cache memory: X GiB
```

If the two numbers don't line up, the problem is the override, so change the override. Only when the two numbers are close are you genuinely short, and at that point the only levers you have are lowering F, cutting the transient budgets, or lowering `--max-model-len` (reasons in "Memory traps on 24GB cards" in README.md).

I misdiagnosed this twice, and both times it wasted a boot.

### A couple hundred MB of memory goes missing, with no warning

Check whether `PYTORCH_CUDA_ALLOC_CONF` contains both `expandable_segments:True` and `max_split_size_mb`. When both are used together, expandable gets silently disabled: PyTorch prints no warning and leaves no trace in the log. The only symptom is more fragmentation, an available figure a notch lower than last time, and then an OOM on some long request.

There is no log line to grep, so read the process environment directly:

```bash
tr '\0' '\n' < /proc/$(pgrep -f "vllm[ ]serve" | head -1)/environ | grep PYTORCH_CUDA_ALLOC_CONF
```

The correct value is that single item on its own:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### The first request after a reboot is twice as slow, and the second run is normal

This is Triton JIT, not a hardware problem and not a config regression.

The numbers: same 200k needle, 72 seconds cold and 44 seconds once warm; decode 22 tok/s cold, 49 warm. Just about exactly half, large enough to make you think the config is broken.

The reason is that the compilation buckets fall along the dimension of "how many prefill chunks one request consumes." Each new chunk-count bucket has to be compiled once, 7-11.5 seconds each time, and it is per-process. Within one bucket it's shared, which is why 10668 and 10752 differ by only 0.87 seconds; cross a bucket and you pay again.

The engine's built-in warmup (`VLLM_DSV4_WARMUP`, on by default) walks the chunk-count ladder at boot. Confirm this line in the log:

```
DSV4 SM8x JIT warmup: mixed dummy runs at token sizes [...]
```

But it doesn't cover everything. Measured, after boot you still pay the tuition once per new length, so the deployment SOP is to fire another round of warmup requests through the HTTP layer after boot, before putting traffic on it:

```bash
python3 deploy/warmup.py --base-url http://127.0.0.1:8000
```

When measuring yourself, sweep twice and take the second sweep. The first sweep's numbers are not real performance, and writing them into a report only misleads you.

This one is easy to misread as a memory failure, because that is what it looks like from the client: a cold 100k prefill pays the compile tax on top of a half-speed run, blows past a 60-second client timeout, and reads as a hang. Check the log for `JIT compilation during inference` before you go looking for an OOM that isn't there.

### Decode is about half the number in the README, and the GPUs look fine

Check the host CPU before anything else. Decode here is kernel-launch bound, not GPU bound, so decode throughput tracks your host's single-thread speed rather than your cards.

The symptom is specific: decode lands near half, prefill is only mildly off, both are flat across context length, and the GPUs are innocent — full boost clock, memory clock at the P2 ceiling, no throttle flag active, and `utilization.gpu` in the 80s while `utilization.memory` sits under 10%. That last pair is the tell. The SMs are occupied but barely moving data, because they are waiting on the host to hand them the next kernel.

Measured on two 8x3090 machines running the identical commit and configuration:

| host | per-kernel launch | decode | prefill @135k |
|---|---|---|---|
| Zen3 desktop, 8 cores | 4.21 us | 51.1 tok/s | 3032 |
| Zen2 server, 64 cores | 8.21 us | 23.4 tok/s | 2436 |

More cores does not help; kernel launch is one thread. The cost scales inversely with core clock, and CUDA graph replay does not — forcing the same machine to 1500 MHz took launch to 16.17 us while graph replay stayed at ~1.13 us. That is also why disabling graphs is so expensive here: 42.8 ms per token with them, 112.5 ms without.

To measure your own host, time a large batch of trivial kernels eagerly and then the same batch captured in a graph. If the eager number is much worse than the graphed one, the host is your ceiling.

There is a lever, with a real price. vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH` for `DeepseekV4ForCausalLM`, which sets `CompilationMode.NONE` and turns inductor off, so nothing gets fused and decode launches thousands of kernels per token. Setting it to `0` turns inductor back on:

```
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

On the slow host that is +126% decode (23.4 -> 52.9 tok/s) and +32% prefill, verified correct on a needle out to 258k tokens. On the fast host it is +4% decode and *-3.8%* prefill, because there was little launch overhead left to remove.

It costs about 1.2 GiB per GPU. On 24 GiB cards that is mutually exclusive with PROFILE-CACHE at 262144: the pool is 650 blocks of admission reservation plus 256 blocks of context, and there is nothing to give back. Measured, it boots, serves short requests, and dies on a long prefill. Raising `--gpu-memory-utilization` does not rescue it — the pool was never the binding constraint, the peak activation headroom was. PROFILE-P8 reserves ~3 blocks instead of 650 and so has the room on paper, but that combination has not been measured.

Not verified with this flag: `deploy/verify/p11_t1_equality.py`, the cache-hit-versus-recompute token identity check. Treat inductor as a throughput option that has passed needle and arithmetic checks, not as a validated configuration.

### OOM hang when sweeping at 190k and above

The transient budgets stack up and blow out. Two knobs to cut directly:

```
VLLM_DSV4_DELTA_GATHER_BUDGET_MB=256
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
```

The measured cost below 128k is about 1% (5194 against 5231 tok/s unpruned), and it buys turning the 253952 cell from a guaranteed failure into a stable pass. These two values are caps on transient workspaces, not on the pool; cutting them does not move `GPU KV cache size`.

### Free memory sits at single-digit MiB under sustained load, and there is nothing left to trim

Measured on this stack: through a 25-request multi-turn agent benchmark, free memory on the tightest GPU bottomed out at **2 MiB**. It did not OOM, but two megabytes is not a margin.

The first two things you will try both make it worse, and both fail loudly at boot rather than silently:

**Do not "fix" the uneven pipeline partition.** vLLM logs `Hidden layers were unevenly partitioned: [22,21]` and helpfully names `VLLM_PP_LAYER_PARTITION`. Setting `21,22` refuses to boot. This model alternates CSA (`compress_ratio` 4) and HCA (`compress_ratio` 128) layers whose KV footprints differ by 32x, so balancing *layer count* is not balancing *memory* — moving one layer changes a stage's `block_stride` by one page and the binding stage flips. The refusal is a 0.02% miss, not a wall: `[21,22]` boots at `--num-gpu-blocks-override 1010` and is worth about +72 MiB on the tight side, because it also swaps which stage is tight. It is a swap with a small remainder, not a reclaim.

**`--num-gpu-blocks-override` is not a cap, it is the answer to the startup check.** With it set, `available_memory` is replaced by `override * bytes_per_block` (`v1/core/kv_cache_utils.py`, the override branch), and that is what gets compared against the memory one max-length request needs. At 262144 the requirement is 0.44 GiB and 1000 blocks supplies 0.45 GiB — a 2% margin. Lowering the override to 930 does not free memory, it fails the check. There is no slack in that number to reclaim.

What actually moves is the workspace arena. It grows to the largest request any caller ever makes and is then frozen for the process lifetime, so whatever pins it high stays pinned. Two knobs cut it:

```
VLLM_DSV4_INDEXER_PREFILL_BUFFER_TOKENS=1048576   # = max_num_seqs * max_model_len
VLLM_DSV4_PREFILL_CHUNK_SIZE=2
```

Both are needed: the arena takes the max of its callers, so cutting the second while the first still pins 331 MiB buys nothing. Together they take the locked arena from **331.00 MB to 129.75 MB**. Confirm it in one boot with `VLLM_DEBUG_WORKSPACE=1` and read the `[WORKSPACE DEBUG] Workspace locked. Current sizes:` line; if it is not 129.75 the same logger names whichever caller became the new ceiling.

Measured end to end on 8x3090 at 262144 with prefix caching, tight-GPU minimum free through the agent benchmark:

| | tight-GPU min free | decode | prefill 18k / 60k / 135k |
|---|---|---|---|
| baseline | 2 MiB | 51.2 | 3327 / 3276 / 3022 |
| + transient caps (`DELTA_GATHER_BUDGET_MB=128`, `MAX_LOGITS_MB=48`, `--max-num-seqs 2`) | 90 MiB | 51.06 | 3262 / 3151 / 2988 |
| + the two arena knobs | 270 MiB | 51.26 | 3221 / 3133 / 2946 |
| + `VLLM_PP_LAYER_PARTITION=21,22` and blocks 1010 | **362 MiB** | 50.83 | 3273 / 3212 / 2998 |

Retrieval was re-verified at the end state: 40/40 cells cold and warm, zero cold-versus-warm cell mismatches. Note that the last row is measured at `--max-num-seqs 2`; the launch profiles ship 4, and that combination has not been measured.

### Two concurrent requests, decode drops to 5 tok/s

First check whether `--long-prefill-token-threshold` is greater than half of `--max-num-batched-tokens`. When the two are equal, one long prefill eats the entire per-step token budget, and the second request can't get in until the first one's prefill finishes; you end up with "one decoding while the other prefills," and a scheduler step filled by prefill only advances decode by one token — measured 4.2-4.8 tok/s, against 50 for a single stream.

This has nothing to do with memory (kv usage was only 49% when it happened) and nothing to do with F (F=512 with override 1200 shows exactly the same symptom).

```bash
curl -s http://127.0.0.1:8000/metrics | grep vllm:kv_cache_usage_perc
```

If that value isn't high and the second request still can't get in, it's the admission reservation keeping it out rather than actual usage; looking at usage will mislead you. The full mechanism and the length boundary are in "Concurrency: at 262144 it is single stream only, and that is structural" in README.md.

### Running the same prompt twice gives different output

This is expected behavior, not a defect, and it has nothing to do with prefix caching. The root cause is a top-k tie race upstream; see "No guarantee of run-to-run reproducibility" in README.md for details. When doing A/B comparisons, pin this variable down with `VLLM_SM86_DET_TOPK=1`, but don't turn it on in production (it's slow, and it isn't the "right answer" either).

### AssertionError at engine init, message about block size divisibility

```
Each KV cache group's real block_size must be divisible by hash_block_size.
block_sizes=[...], hash_block_size=...
```

`hash_block_size` defaults to the GCD of the group block sizes, and on the GCD side dcp_exempt groups must use the size **without the dcp multiplier**. In this configuration the manager's real sizes are `[1024, 64, 4, 8]`, the GCD is 4, and the assertion holds; the moment an exempt group gets multiplied by dcp while the GCD is being computed, the GCD becomes 16 and `4 % 16` blows up on the spot. Inside the branch the dcp_exempt decision has been unified across the scheduler side and the manager side (that is P11's B2), so a normal configuration should not hit this.

Which means hitting it tells you some size really was changed — `--decode-context-parallel-size`, `--block-size`, or the compression ratio. `--prefix-match-unit` lets you set this value by hand, but it only accepts numbers that divide **every** group block size, which here caps it at 4 (that is, the default; 1 and 2 are legal but finer, they just make the block hash run more times). There is no room to route around it upward, and the `block_sizes` set in the message is the thing to look at.

### Boot is refused, message says PROFILE-P8 and PROFILE-CACHE are mutually exclusive

You have set both `VLLM_DSV4_COMPRESSOR_WINDOWED=1` and `--enable-prefix-caching`. This is a deliberate hard refusal with no silent downgrade — the KV footprints of the two layouts differ by gigabytes, and silently picking one would only come back much later as an admission rejection or an OOM. Pick one configuration per the message; reasons in "Choosing between the two profiles" in README.md.

One thing to watch here: this message suggests holding `max_num_batched_tokens` down to 512 under PROFILE-CACHE, which was the conservative value at the time the guard was written. The ceiling that later measurement produced is 768 (see the memory accounting in "Choosing between the two profiles" in README.md), and 768 is what to go by. The message itself has not been updated.

### Prefill lands ~15% below every table in the README, uniformly across lengths

If the numbers are shifted down by a near-constant ratio at every length while decode is untouched, check the commit range before touching any knob: dc5487ef2 (the cross-request zeroing coverage fix) initially cost a flat ~40 µs/token. The fp32 compressor-state groups allocate blocks at 4-8 token granularity — ~290 ids per F=768 step — and each id was zeroed through a launch grid of every segment times the largest segment's chunk count: millions of mostly early-exiting thread blocks per step. 51783a09b (per-group zeroing) recovers it with identical coverage; the tables reproduce at or past that commit.

The same tax is why F stopped mattering inside that window: a flat per-token cost compresses the differential between fast configurations, so F=512→768 measured +7% instead of the table's +26%. If raising F barely moves prefill, suspect an added flat per-step or per-token cost before suspecting the batching path itself — the giveaway is a deficit ratio that is identical at every length.

Boot prints a one-line census of the zeroer's segment tables (`KVBlockZeroer segments: flat=N per-group={...}`). Every group that allocates blocks must appear with a plausible count; on this configuration the per-group sum exceeds the flat count, because the packed layout aliases every group's first layer to one address and that segment appears once per group. A missing group there means zeroing coverage was silently lost at init — stop for that; it is a correctness hole, not a perf note.
