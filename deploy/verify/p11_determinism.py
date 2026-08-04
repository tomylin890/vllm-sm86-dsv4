#!/usr/bin/env python3
"""Isolate the source of run-to-run nondeterminism (caching NOT involved).

Same prompt, N consecutive serial requests, temperature 0. Reports whether all
N token streams are identical and where they first diverge. Run it on config C
(caching OFF) with one suspect flag flipped per boot:

    baseline                      -> reproduces the divergence
    VLLM_DSV4_DELTA_GATHER=0      -> P7 staging lifecycle
    VLLM_DSV4_FLASH_DECODE=0      -> flash-mla split-K decode path

Divergence in the FIRST decode step implicates prefill/indexer; divergence
later implicates the decode path.
"""
import sys
from p11lib import Client, build_ids, encode_fragment, seed_of

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

c = Client(verbose=True)
tail = encode_fragment(c, "\n\n請根據上文,用一句話總結。")
body = build_ids(c, N - len(tail), seed_of("det", N))
ids = body + tail
runs = []
for i in range(REPS):
    r = c.complete(ids, 64, seed=7)
    runs.append(r.get("token_ids") or [])
    print(f"run{i}: ntok={len(runs[-1])} ttft={r.get('ttft_s') and round(r['ttft_s'],2)}s")
base = runs[0]
alldiv = []
for i, r in enumerate(runs[1:], 1):
    div = next((k for k, (x, y) in enumerate(zip(base, r)) if x != y), None)
    alldiv.append(div)
    print(f"run0 vs run{i}: {'IDENTICAL' if div is None else f'div@{div}'}")
print("VERDICT:", "DETERMINISTIC" if all(d is None for d in alldiv)
      else f"NONDETERMINISTIC (first divergences: {alldiv})")
