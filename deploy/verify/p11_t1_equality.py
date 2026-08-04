#!/usr/bin/env python3
"""T1 -- cache-hit vs recompute EXACT token equality (the gating test).

Two-phase, because B and C are separate boots of one machine:

  run phase (on the rack, against the live server):
      python3 p11_t1_equality.py run --config B --out t1_B.json
      python3 p11_t1_equality.py run --config C --out t1_C.json
  compare phase (anywhere):
      python3 p11_t1_equality.py compare --b t1_B.json --c t1_C.json

Per prompt length {8k, 32k, 128k, 200k}: turn-1 (cold; on B it populates the
cache) then turn-2 (same token-ID prompt; B = warm hit, C = cold recompute).
Prompts are token-ID arrays built server-side from a fixed seed, so both
configs see byte-identical inputs by construction. Requests are STRICTLY
serial -- one in flight, ever -- which keeps batch composition identical
across runs (vLLM is batch-variant; serial submission is the control).

Acceptance (design T1): 100%% token identity B-vs-C at turn 1 (isolates
ring-off from caching-on) AND at turn 2 (warm vs cold). C.turn1 == C.turn2
is the rig-determinism control. Logprob deltas on the first 16 decode steps
are recorded as a seam diagnostic, not a criterion; divergence at index 0
specifically implicates the compressor seam.

On B, /reset_prefix_cache runs before each case so turn-1 is provably cold
(the boot warmup sweep populated the cache with its own prompts).
"""

import argparse
import json
import os
import subprocess
import sys
import time

from p11lib import Client, PromptCache, build_ids, seed_of, write_json

LENGTHS = tuple(int(x) for x in os.environ.get(
    "P11_T1_LENGTHS", "8192 32768 131072 200704").split())
MAX_TOKENS = 256
CASE_PAUSE_S = 20


def gpu_temps():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return [int(x) for x in out.split()]
    except Exception:  # noqa: BLE001 - temps are telemetry, never fatal
        return []


def run_phase(args):
    client = Client(base_url=args.base_url, verbose=True)
    cache = PromptCache(args.prompt_cache)
    out = {"config": args.config, "base_url": args.base_url,
           "model": client.model(), "started": time.time(), "cases": []}
    for length in LENGTHS:
        ids = build_ids(client, length, seed_of("t1", length),
                        cache=cache, key=f"t1-{length}")
        if args.config == "B":
            client.reset_prefix_cache()
            time.sleep(2)
        case = {"length": length, "temps_before": gpu_temps()}
        for turn in (1, 2):
            rec = client.complete(ids, MAX_TOKENS, logprobs=1,
                                  seed=seed_of("t1", length, "sample"))
            rec["turn"] = turn
            case[f"turn{turn}"] = rec
            status = "OK" if rec["ok"] else f"ERR:{rec.get('error_kind')}"
            print(f"[t1:{args.config}] len={length} turn={turn} {status} "
                  f"ttft={rec.get('ttft_s') and round(rec['ttft_s'], 2)}s "
                  f"cached={rec.get('cached_tokens')} "
                  f"ntok={len(rec.get('token_ids') or [])}", flush=True)
        case["temps_after"] = gpu_temps()
        out["cases"].append(case)
        write_json(args.out, out)  # persist after every case: crash-safe
        time.sleep(CASE_PAUSE_S)
    write_json(args.out, out)
    print(f"[t1:{args.config}] wrote {args.out}")


def _identity(name, a, b, failures):
    ta, tb = a.get("token_ids") or [], b.get("token_ids") or []
    if not a.get("ok") or not b.get("ok"):
        failures.append(f"{name}: unusable record "
                        f"(a_ok={a.get('ok')} b_ok={b.get('ok')})")
        return
    if ta == tb:
        print(f"  PASS  {name}: {len(ta)} tokens identical")
        return
    div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
               min(len(ta), len(tb)))
    msg = (f"{name}: DIVERGED at decode index {div} "
           f"(len a={len(ta)} b={len(tb)})")
    if div == 0:
        msg += " -- index 0: implicates the compressor seam (I1/I6)"
    failures.append(msg)
    print(f"  FAIL  {msg}")


def _logprob_delta(a, b, n=16):
    la, lb = a.get("token_logprobs") or [], b.get("token_logprobs") or []
    pairs = [(x, y) for x, y in list(zip(la, lb))[:n]
             if x is not None and y is not None]
    if not pairs:
        return None
    return max(abs(x - y) for x, y in pairs)


def compare_phase(args):
    b = json.load(open(args.b))
    c = json.load(open(args.c))
    failures = []
    for cb, cc in zip(b["cases"], c["cases"]):
        assert cb["length"] == cc["length"], "case order mismatch"
        length = cb["length"]
        print(f"case len={length}:")
        _identity(f"len={length} T1a turn1 B==C", cb["turn1"], cc["turn1"],
                  failures)
        _identity(f"len={length} T1b turn2 B(warm)==C(cold)", cb["turn2"],
                  cc["turn2"], failures)
        _identity(f"len={length} determinism control C1==C2", cc["turn1"],
                  cc["turn2"], failures)
        cached = cb["turn2"].get("cached_tokens")
        expect = (length // 1024) * 1024
        if cached is not None:
            mark = "PASS" if cached >= expect - 1024 and cached > 0 else "FAIL"
            print(f"  {mark}  warm hit evidence: cached_tokens={cached} "
                  f"(expect ~{expect})")
            if mark == "FAIL":
                failures.append(f"len={length}: cached_tokens={cached}, "
                                f"expected ~{expect} -- caching not engaged")
        else:
            print("  INFO  usage carries no cached_tokens; rely on TTFT + "
                  "metrics for hit evidence")
        d = _logprob_delta(cb["turn2"], cc["turn2"])
        if d is not None:
            print(f"  INFO  max|dlogprob| first 16 steps: {d:.6g}")
        t_b, t_c = cb["turn2"].get("ttft_s"), cc["turn2"].get("ttft_s")
        if t_b and t_c:
            print(f"  INFO  turn2 TTFT B={t_b:.2f}s vs C={t_c:.2f}s")
    print()
    if failures:
        print(f"T1 VERDICT: FAIL ({len(failures)} problem(s))")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("T1 VERDICT: PASS -- 100% token identity, gate open")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", choices=("B", "C"), required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--prompt-cache", default="t1_prompt_cache.json")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--c", required=True)
    args = ap.parse_args()
    if args.phase == "run":
        run_phase(args)
    else:
        compare_phase(args)


if __name__ == "__main__":
    main()


