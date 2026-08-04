#!/usr/bin/env python3
"""Cold prefill throughput at a few lengths. Every point uses a FRESH filler
seed, so nothing hits the prefix cache and the number is true prefill."""
import sys
from p11lib import Client, build_ids, seed_of

LENGTHS = [int(x) for x in (sys.argv[1:] or
                            ["16384", "65536", "131072", "200704", "253952"])]
c = Client(verbose=True)
salt = "probe-%d" % len(LENGTHS)
print("%9s %10s %12s" % ("tokens", "TTFT", "prefill tok/s"))
for n in LENGTHS:
    ids = build_ids(c, n, seed_of("prefillprobe", salt, n))
    r = c.complete(ids, 8, seed=5)
    t = r.get("ttft_s")
    print("%9d %9.2fs %12.0f%s" % (n, t or -1, (n / t) if t else -1,
                                   "" if r.get("ok") else "  ERR"))
