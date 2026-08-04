#!/usr/bin/env python3
"""Headline check: a near-max-context request under PROFILE-CACHE.

mml is 262144 and the reply needs room, so the prompt is 261632 (= 255.5 KiB
tokens) + 64 generated. Cold then warm, needle at mid depth, thinking OFF.
This is the single request the whole P11 exercise exists to make possible.
"""
import sys
from p11lib import Client, build_ids, encode_fragment, seed_of

N = int(sys.argv[1]) if len(sys.argv) > 1 else 261632
WORD = "LANTERN"

c = Client(verbose=True)
prefix, suffix = c.chat_wrapper(thinking=False)
needle = encode_fragment(c, f"\n請記住:通關密語是 {WORD}。\n")
q = encode_fragment(c, "\n\n上文中的通關密語是什麼?只回答那個英文單字。")
body = build_ids(c, N - len(prefix) - len(suffix) - len(needle) - len(q),
                 seed_of("maxctx", N))
half = len(body) // 2
ids = prefix + body[:half] + needle + body[half:] + q + suffix
assert len(ids) == N, len(ids)
for tag in ("COLD", "WARM"):
    m0 = c.metrics()
    r = c.complete(ids, 64, seed=99)
    m1 = c.metrics()
    hits = int(c.metric(m1, "vllm:prefix_cache_hits") - c.metric(m0, "vllm:prefix_cache_hits"))
    txt = (r.get("text") or "")
    print(f"{tag}: n={N} ok={r.get('ok')} pass={WORD in txt.upper()} "
          f"ttft={r.get('ttft_s') and round(r['ttft_s'], 2)}s hits={hits} "
          f"text={txt[:60]!r}")
