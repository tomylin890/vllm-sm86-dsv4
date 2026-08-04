#!/usr/bin/env python3
"""Power-limit A/B: what does capping the cards at 280 W actually cost?

Runs against a STANDING server -- the power limit takes effect immediately, so
both halves of the A/B run on the same boot, the same weights and the same
cache state. That removes boot-to-boot variance, which on this stack is large
enough (Triton JIT buckets, allocator layout) to swamp a 10%% effect.

Expected shape of the result, stated up front so the measurement can refute it:
prefill is compute-bound and should take most of the hit; decode at batch 1 is
bound by memory bandwidth and kernel latency, and GDDR6X bandwidth does not
scale down with core power, so decode should barely move.

  python3 p11_pl_ab.py                    # 350 -> 280 -> restore
  python3 p11_pl_ab.py --watts 250 300    # sweep several limits
  python3 p11_pl_ab.py --no-restore       # leave the last limit in place

THERMAL PACING is not optional here: the PSU on this box trips on sustained
eight-card load, and reducing that is the whole point of the experiment. Every
round is followed by a cooldown that waits BOTH a fixed time and for the GPUs
to return to the run's baseline temperature.

SUDO: `nvidia-smi -pl` needs root. This script never handles a password. It
tries `sudo -n` (passwordless); if that is not configured it prints the exact
command and waits for you to run it in another shell, then continues.
"""

import argparse
import json
import subprocess
import sys
import time

from p11lib import Client, build_ids, seed_of, write_json

LENGTHS = (16384, 65536, 131072, 200704)
GEN_TOKENS = 96


def smi(query, extra=()):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
             *extra],
            capture_output=True, text=True, timeout=15).stdout
        return [float(x) for x in out.split()]
    except Exception:  # noqa: BLE001 - telemetry must never abort a run
        return []


def temps():
    return smi("temperature.gpu")


def power():
    return smi("power.draw")


def current_limit():
    vals = smi("power.limit")
    return vals[0] if vals else None


def default_limit():
    vals = smi("power.default_limit")
    return vals[0] if vals else None


def set_limit(watts):
    """Returns True if the limit is now `watts` on every card."""
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(int(watts))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("\n  passwordless sudo is not available. Run this in another "
              "shell, then press Enter here:\n")
        print(f"      sudo nvidia-smi -pl {int(watts)}\n")
        try:
            input("  [Enter] once the command has completed: ")
        except EOFError:
            print("  no tty -- cannot continue interactively")
            return False
    time.sleep(2)
    got = smi("power.limit")
    ok = bool(got) and all(abs(g - watts) < 1.5 for g in got)
    print(f"  power limit now: {got} ({'OK' if ok else 'MISMATCH'})")
    return ok


def cooldown(baseline, seconds, margin=5.0):
    start = time.monotonic()
    print(f"  [thermal] cooling >= {seconds}s and until max temp <= "
          f"{baseline + margin:.0f}C", flush=True)
    time.sleep(seconds)
    for _ in range(80):
        t = temps()
        if not t or max(t) <= baseline + margin:
            break
        time.sleep(15)
    return time.monotonic() - start


def measure(client, tag, salt):
    """One pass over LENGTHS: cold prefill + steady decode, with power sampled
    during the prefill (the compute-heavy phase the limit actually bites)."""
    rows = []
    for n in LENGTHS:
        ids = build_ids(client, n, seed_of("plab", salt, n))
        p_before = power()
        rec = client.complete(ids, GEN_TOKENS, seed=17)
        p_after = power()
        t = rec.get("ttft_s")
        row = {
            "tokens": n,
            "ttft_s": t,
            "prefill_tok_s": (n / t) if t else None,
            "decode_tok_s": rec.get("decode_tok_s"),
            "ok": rec.get("ok"),
            "power_w_sum": round(sum(p_before + p_after) / 2, 1)
            if p_before and p_after else None,
            "temps": temps(),
        }
        rows.append(row)
        print(f"  [{tag}] {n:>7} tok  ttft={row['ttft_s'] and round(row['ttft_s'],2)}s "
              f"prefill={row['prefill_tok_s'] and round(row['prefill_tok_s'])} tok/s "
              f"decode={row['decode_tok_s'] and round(row['decode_tok_s'],1)} "
              f"maxT={max(row['temps']) if row['temps'] else '-'}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watts", type=float, nargs="*", default=[280.0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="pl_ab.json")
    ap.add_argument("--cooldown-seconds", type=float, default=180)
    ap.add_argument("--no-restore", action="store_true")
    args = ap.parse_args()

    client = Client(base_url=args.base_url, verbose=True)
    baseline_t = max(temps() or [40])
    start_limit = current_limit()
    restore_to = default_limit() or start_limit
    print(f"current power limit {start_limit} W, card default {restore_to} W, "
          f"baseline temp {baseline_t:.0f}C")

    out = {"default_limit_w": restore_to, "start_limit_w": start_limit,
           "baseline_temp_c": baseline_t, "passes": []}
    try:
        print(f"\n=== pass at {start_limit:.0f} W (as found) ===")
        out["passes"].append({"watts": start_limit,
                              "rows": measure(client, f"{start_limit:.0f}W", "a")})
        write_json(args.out, out)

        for i, w in enumerate(args.watts):
            cooldown(baseline_t, args.cooldown_seconds)
            print(f"\n=== setting power limit to {w:.0f} W ===")
            if not set_limit(w):
                print("  could not set the limit -- skipping this point")
                continue
            out["passes"].append({"watts": w,
                                  "rows": measure(client, f"{w:.0f}W", f"b{i}")})
            write_json(args.out, out)
    finally:
        if not args.no_restore and restore_to:
            print(f"\n=== restoring power limit to {restore_to:.0f} W ===")
            set_limit(restore_to)

    base = out["passes"][0]
    print(f"\n{'tokens':>8} " + " ".join(f"{p['watts']:.0f}W prefill  " for p in out["passes"])
          + "  |  " + " ".join(f"{p['watts']:.0f}W decode " for p in out["passes"]))
    for i, r0 in enumerate(base["rows"]):
        pre = " ".join(
            f"{(p['rows'][i]['prefill_tok_s'] or 0):>9.0f}      " for p in out["passes"])
        dec = " ".join(
            f"{(p['rows'][i]['decode_tok_s'] or 0):>8.1f}    " for p in out["passes"])
        print(f"{r0['tokens']:>8} {pre}  |  {dec}")
    for p in out["passes"][1:]:
        dp = [(p["rows"][i]["prefill_tok_s"] or 0) / (base["rows"][i]["prefill_tok_s"] or 1)
              for i in range(len(base["rows"]))]
        dd = [(p["rows"][i]["decode_tok_s"] or 0) / (base["rows"][i]["decode_tok_s"] or 1)
              for i in range(len(base["rows"]))]
        print(f"\n{p['watts']:.0f} W vs {base['watts']:.0f} W: "
              f"prefill {min(dp) * 100:.0f}-{max(dp) * 100:.0f}% of baseline, "
              f"decode {min(dd) * 100:.0f}-{max(dd) * 100:.0f}%")
        pw_b = [r["power_w_sum"] for r in base["rows"] if r["power_w_sum"]]
        pw_p = [r["power_w_sum"] for r in p["rows"] if r["power_w_sum"]]
        if pw_b and pw_p:
            print(f"  board power (8 cards, sampled): "
                  f"{sum(pw_b)/len(pw_b):.0f} W -> {sum(pw_p)/len(pw_p):.0f} W")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
