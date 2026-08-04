#!/usr/bin/env python3
"""Why is concurrency-2 slow with caching on? Sample the scheduler while two
requests are in flight.

If num_requests_running never reaches 2 while num_requests_waiting is 1, the
engine is SERIALISING them -- a capacity decision, not a performance bug. The
suspected cause is the sliding-window reservation: with the compressor-state
ring off, each long request reserves ~C(F) blocks (~650 at F=768), so two of
them need ~1300 against a 1000-block pool.
"""
import concurrent.futures as cf
import sys
import threading
import time

from p11lib import Client, build_ids, seed_of

N = int(sys.argv[1]) if len(sys.argv) > 1 else 61952
c = Client(verbose=False)
stop = threading.Event()
samples = []


def sampler():
    while not stop.is_set():
        try:
            m = c.metrics()
            samples.append((
                round(time.monotonic() % 10000, 1),
                int(c.metric(m, "vllm:num_requests_running")),
                int(c.metric(m, "vllm:num_requests_waiting")),
                round(c.metric(m, "vllm:kv_cache_usage_perc"), 3)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)


def one(i):
    ids = build_ids(c, N, seed_of("conc", N, i))
    t0 = time.monotonic()
    r = c.complete(ids, 128, seed=11 + i)
    return i, r.get("ttft_s"), time.monotonic() - t0, len(r.get("token_ids") or [])


th = threading.Thread(target=sampler, daemon=True); th.start()
with cf.ThreadPoolExecutor(max_workers=2) as pool:
    futs = [pool.submit(one, i) for i in range(2)]
    res = [f.result() for f in futs]
stop.set(); time.sleep(0.6)

for i, ttft, total, ntok in sorted(res):
    print("req%d: ttft=%.2fs total=%.2fs ntok=%d decode=%.1f tok/s" % (
        i, ttft or -1, total, ntok, (ntok - 1) / (total - ttft) if ttft and total > ttft else -1))
run_max = max((s[1] for s in samples), default=-1)
wait_max = max((s[2] for s in samples), default=-1)
kv_max = max((s[3] for s in samples), default=-1)
both = sum(1 for s in samples if s[1] >= 2)
print("samples=%d  max_running=%d  max_waiting=%d  max_kv_usage=%.3f  steps_with_2_running=%d"
      % (len(samples), run_max, wait_max, kv_max, both))
print("VERDICT:", "SERIALISED (capacity)" if run_max < 2 else "genuinely concurrent")
print("trace(running,waiting,kv):", [(s[1], s[2], s[3]) for s in samples[::max(1, len(samples)//14)]][:15])
