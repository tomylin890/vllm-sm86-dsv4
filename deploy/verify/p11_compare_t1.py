#!/usr/bin/env python3
"""T1 verdict with the determinism control that the original gate lacked.

  python3 p11_compare_t1.py --b t1_B.json --c t1_C.json

The design's gate was "100%% token identity, B warm vs C cold". That is only a
meaningful gate if the stack is bit-reproducible in the first place. Config C
(caching OFF) runs the SAME prompt twice, so C.turn1 vs C.turn2 measures the
stack's inherent run-to-run variance with no cache involved. Read the three
numbers together:

  C1 vs C2  = inherent nondeterminism (no caching anywhere)
  B1 vs B2  = cold vs warm on the caching build
  B2 vs C2  = warm output vs an uncached reference

If C1 vs C2 already diverges, exact identity is unachievable on this stack and
the honest acceptance is "caching adds no divergence beyond the inherent
baseline" -- judged by first-divergence index and logprob deltas, plus the
semantic gates (T2 needle retrieval with thinking off).
"""

import argparse
import json


def cmp_pair(a, b):
    ta, tb = a.get("token_ids") or [], b.get("token_ids") or []
    if not (a.get("ok") and b.get("ok")):
        return {"usable": False}
    div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    la, lb = a.get("token_logprobs") or [], b.get("token_logprobs") or []
    # Only positions BEFORE the first divergence are comparable: past it the
    # two runs are on different trajectories and their logprobs describe
    # different tokens. `div if div else len(la)` was wrong -- div == 0 is
    # falsy, so a divergence at the very first token silently compared the
    # whole (incomparable) tail.
    upto = len(la) if div is None else div
    pre = [abs(x - y) for x, y in list(zip(la, lb))[:upto]
           if x is not None and y is not None]
    return {
        "usable": True,
        "identical": ta == tb,
        "first_div": div,
        "n": min(len(ta), len(tb)),
        "n_comparable": len(pre),
        "max_dlp_before_div": max(pre) if pre else None,
    }


def fmt(r):
    if not r.get("usable"):
        return "UNUSABLE"
    if r["identical"]:
        return f"identical ({r['n']} tok)"
    d = r["max_dlp_before_div"]
    nc = r.get("n_comparable", 0)
    return (f"div@{r['first_div']}/{r['n']}"
            + (f" max|dlp|={d:.4f} over {nc}" if d is not None
               else " (no comparable positions)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", required=True)
    args = ap.parse_args()
    b = json.load(open(args.b))
    c = json.load(open(args.c))
    cmap = {x["length"]: x for x in c["cases"]}

    print(f"{'len':>8}  {'C1 vs C2 (control)':<34} {'B1 vs B2 (cold vs warm)':<34} "
          f"{'B2 vs C2 (warm vs uncached)':<34}")
    control_clean = True
    caching_clean = True
    for case in b["cases"]:
        n = case["length"]
        cc = cmap.get(n)
        if not cc:
            continue
        ctl = cmp_pair(cc["turn1"], cc["turn2"])
        bb = cmp_pair(case["turn1"], case["turn2"])
        cross = cmp_pair(case["turn2"], cc["turn2"])
        control_clean &= bool(ctl.get("identical"))
        caching_clean &= bool(bb.get("identical"))
        print(f"{n:>8}  {fmt(ctl):<34} {fmt(bb):<34} {fmt(cross):<34}")

    print()
    print(f"  TTFT (B cold -> B warm):")
    for case in b["cases"]:
        t1 = case["turn1"].get("ttft_s")
        t2 = case["turn2"].get("ttft_s")
        cc = cmap.get(case["length"], {})
        tc = (cc.get("turn2") or {}).get("ttft_s")
        if t1 and t2:
            print(f"    len={case['length']:>7}: {t1:>6.2f}s -> {t2:>5.2f}s "
                  f"({t1 / t2:>5.1f}x)   C reference {tc and round(tc, 2)}s")
    print()
    if control_clean and caching_clean:
        print("VERDICT: stack is bit-reproducible AND caching preserves it.")
    elif control_clean and not caching_clean:
        print("VERDICT: control is bit-reproducible but the CACHED path is not "
              "-- caching introduces the divergence. Investigate before ship.")
    elif not control_clean:
        print("VERDICT: the stack is NOT bit-reproducible with caching OFF, so "
              "exact token identity was never an achievable gate. Judge caching "
              "by whether its divergence is of the same order as the control "
              "(above) and by the semantic gates (T2/T3).")


if __name__ == "__main__":
    main()


