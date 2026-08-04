#!/usr/bin/env python3
"""T2 -- needle retrieval from a CACHED prefix, per-cell cold/warm/control.

  run phase:
      python3 p11_t2_needle.py run --config B --out t2_B.json   # cold+warm per cell
      python3 p11_t2_needle.py run --config C --out t2_C.json   # cold only
  compare phase:
      python3 p11_t2_needle.py compare --b t2_B.json --c t2_C.json

Grid: depths x contexts. On B every cell runs twice back-to-back: run 1 cold
(populates the cache), run 2 warm (hits floor(n/1024)*1024). On C once.
Acceptance is PER CELL, never aggregate: warm == cold == C. A unique filler
(seed keyed on ctx AND depth) keeps every cold run genuinely cold without
depending on /reset_prefix_cache; a unique per-cell codeword prevents a stale
answer from faking a pass.

THERMAL PACING: after every --round-seconds of active load the harness
pauses --cooldown-seconds AND until all GPU temps fall back to the run
baseline + 5 C (PSU protection discipline). Cooldown time is excluded from
every latency measurement (all timings are per-request).
"""

import argparse
import json
import subprocess
import sys
import time

from p11lib import (
    Client, PromptCache, Rng, build_ids, encode_fragment, seed_of, write_json,
)

DEPTHS = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95)
CONTEXTS = (32768, 65536, 131072, 200704, 253952)
MAX_TOKENS = 64  # thinking is OFF, so the answer is the first thing emitted
WORDS = ("KOALA", "ZEBRA", "MANGO", "COMET", "PIANO", "TIGER", "OASIS",
         "RIVER", "CANDLE", "FALCON", "MARBLE", "ORCHID", "PUZZLE", "SADDLE",
         "TROPHY", "VELVET", "WALNUT", "ANCHOR", "BREEZE", "CIRCUS")


def gpu_temps():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return [int(x) for x in out.split()]
    except Exception:  # noqa: BLE001
        return []


class ThermalPacer:
    def __init__(self, round_s: float, cooldown_s: float, margin_c: int = 5):
        self.round_s, self.cooldown_s, self.margin = round_s, cooldown_s, margin_c
        self.baseline = max(gpu_temps() or [0])
        self.load_started = time.monotonic()
        self.pauses = []

    def maybe_cooldown(self):
        if time.monotonic() - self.load_started < self.round_s:
            return
        start = time.monotonic()
        print(f"[thermal] round limit reached; cooling >= {self.cooldown_s}s "
              f"and until max temp <= {self.baseline + self.margin}C", flush=True)
        time.sleep(self.cooldown_s)
        for _ in range(60):
            temps = gpu_temps()
            if not temps or max(temps) <= self.baseline + self.margin:
                break
            time.sleep(15)
        self.pauses.append({"at": time.time(),
                            "waited_s": time.monotonic() - start,
                            "temps": gpu_temps()})
        self.load_started = time.monotonic()


def cell_word(ctx, depth):
    return WORDS[Rng(seed_of("t2word", ctx, depth)).below(len(WORDS))]


def build_cell_ids(client, cache, ctx, depth):
    """Chat-templated, thinking-DISABLED prompt of EXACTLY ``ctx`` tokens.

    Thinking off is deliberate: with reasoning on, a 64-token budget is spent
    inside the think trace and the answer never appears, and any near-tie
    difference between a cached and an uncached run gets amplified by a long
    self-conditioning trace. Retrieval correctness is what T2 measures, so the
    trace is removed rather than measured.
    """
    word = cell_word(ctx, depth)
    prefix, suffix = client.chat_wrapper(thinking=False)
    needle = encode_fragment(client, f"\n請記住:通關密語是 {word}。\n")
    question = encode_fragment(
        client, "\n\n上文中的通關密語是什麼?只回答那個英文單字。")
    body_len = ctx - len(prefix) - len(suffix) - len(needle) - len(question)
    body = build_ids(client, body_len, seed_of("t2", ctx, depth),
                     cache=cache, key=f"t2-{ctx}-{depth}-{body_len}")
    pos = max(1, int(body_len * depth))
    ids = prefix + body[:pos] + needle + body[pos:] + question + suffix
    assert len(ids) == ctx, f"built {len(ids)} tokens, wanted {ctx}"
    return ids, word


def judge(rec, word):
    return rec.get("ok") and word.upper() in (rec.get("text") or "").upper()


def run_phase(args):
    client = Client(base_url=args.base_url, verbose=True)
    cache = PromptCache(args.prompt_cache)
    pacer = ThermalPacer(args.round_seconds, args.cooldown_seconds)
    runs = ("cold", "warm") if args.config == "B" else ("cold",)
    out = {"config": args.config, "model": client.model(),
           "baseline_temps": gpu_temps(), "cells": [], "thermal_pauses": None}
    for ctx in CONTEXTS:
        for depth in DEPTHS:
            pacer.maybe_cooldown()
            ids, word = build_cell_ids(client, cache, ctx, depth)
            cell = {"ctx": ctx, "depth": depth, "word": word,
                    "temps": gpu_temps()}
            for phase in runs:
                rec = client.complete(ids, MAX_TOKENS,
                                      seed=seed_of("t2s", ctx, depth))
                cell[phase] = {
                    "pass": judge(rec, word), "ok": rec.get("ok"),
                    "error_kind": rec.get("error_kind"),
                    "ttft_s": rec.get("ttft_s"),
                    "cached_tokens": rec.get("cached_tokens"),
                    "ntok": len(rec.get("token_ids") or []),
                    "text": (rec.get("text") or "")[:160],
                }
                p = cell[phase]
                print(f"[t2:{args.config}] ctx={ctx} depth={depth} {phase}: "
                      f"{'PASS' if p['pass'] else 'FAIL'} "
                      f"ttft={p['ttft_s'] and round(p['ttft_s'], 2)}s "
                      f"cached={p['cached_tokens']}", flush=True)
            out["cells"].append(cell)
            out["thermal_pauses"] = pacer.pauses
            write_json(args.out, out)
    print(f"[t2:{args.config}] wrote {args.out}")


def compare_phase(args):
    b = json.load(open(args.b))
    c = json.load(open(args.c))
    c_index = {(x["ctx"], x["depth"]): x for x in c["cells"]}
    failures, warm_ttfts = [], []
    print(f"{'ctx':>8} {'depth':>6} {'B cold':>7} {'B warm':>7} {'C':>5} "
          f"{'ttft cold':>10} {'ttft warm':>10} {'cached':>8}")
    for cell in b["cells"]:
        key = (cell["ctx"], cell["depth"])
        cc = c_index.get(key)
        bc, bw = cell.get("cold", {}), cell.get("warm", {})
        cp = cc.get("cold", {}) if cc else {}
        row_ok = bc.get("pass") and bw.get("pass") and cp.get("pass")
        print(f"{cell['ctx']:>8} {cell['depth']:>6} "
              f"{'PASS' if bc.get('pass') else 'FAIL':>7} "
              f"{'PASS' if bw.get('pass') else 'FAIL':>7} "
              f"{'PASS' if cp.get('pass') else 'FAIL':>5} "
              f"{(bc.get('ttft_s') or 0):>9.2f}s {(bw.get('ttft_s') or 0):>9.2f}s "
              f"{bw.get('cached_tokens') if bw.get('cached_tokens') is not None else '-':>8}")
        if not row_ok:
            failures.append(f"ctx={cell['ctx']} depth={cell['depth']}: "
                            f"cold={bc.get('pass')} warm={bw.get('pass')} "
                            f"C={cp.get('pass')}")
        if bc.get("ttft_s") and bw.get("ttft_s"):
            warm_ttfts.append((cell["ctx"], bc["ttft_s"], bw["ttft_s"]))
    if warm_ttfts:
        print("\nwarm-TTFT collapse by context (median cold -> median warm):")
        for ctx in sorted({c for c, _, _ in warm_ttfts}):
            colds = sorted(x for c, x, _ in warm_ttfts if c == ctx)
            warms = sorted(x for c, _, x in warm_ttfts if c == ctx)
            print(f"  ctx={ctx}: {colds[len(colds) // 2]:.2f}s -> "
                  f"{warms[len(warms) // 2]:.2f}s")
    print()
    if failures:
        print(f"T2 VERDICT: FAIL ({len(failures)} cell(s))")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("T2 VERDICT: PASS -- every cell cold==warm==control")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", choices=("B", "C"), required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--prompt-cache", default="t2_prompt_cache.json")
    run.add_argument("--round-seconds", type=float, default=300)
    run.add_argument("--cooldown-seconds", type=float, default=180)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--c", required=True)
    args = ap.parse_args()
    run_phase(args) if args.phase == "run" else compare_phase(args)


if __name__ == "__main__":
    main()


