#!/usr/bin/env python3
"""T3 -- multi-turn agent benchmark: hit-length behavior + TTFT gain.

  python3 p11_t3_agent_bench.py run --config B --out t3_B.json
  python3 p11_t3_agent_bench.py run --config C --out t3_C.json   # or A
  python3 p11_t3_agent_bench.py compare --b t3_B.json --baseline t3_C.json

One session per target context in {8k, 32k, 64k, 128k, 200k}: an 8k system
prompt + filler history padded to the target, then 5 turns. Each turn's
prompt = previous prompt + previous completion + a short new user message --
the canonical agent pattern, all at token-ID level (completions come back as
IDs and are appended verbatim).

Per turn it records TTFT, decode tok/s, usage, and cached_tokens (the
server-reported per-request hit length). Acceptance on B:
  (i)  turn >= 2: cached_tokens is within one 1024-token logical block of
       floor(previous_request_total / 1024) * 1024  (hit-length arithmetic,
       design T3(i); previous_request_total = prev prompt + prev completion);
  (ii) turn-2 TTFT at the 200k session collapses vs the baseline config
       (< 3 s absolute per the design).
The baseline run (A or C) reports the same numbers with caching off.

Thermal pacing: sessions are separated by the round/cooldown gate; the heavy
load is each session's turn-1 prefill, turns 2-5 are incremental.
"""

import argparse
import json
import subprocess
import sys
import time

from p11lib import Client, PromptCache, build_ids, encode_fragment, seed_of, write_json

CONTEXTS = (8192, 32768, 65536, 131072, 200704)
TURNS = 5
REPLY_TOKENS = 200
SYSTEM_TOKENS = 8192
BLOCK = 1024


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


def cooldown(baseline, cooldown_s, margin=5):
    start = time.monotonic()
    time.sleep(cooldown_s)
    for _ in range(60):
        temps = gpu_temps()
        if not temps or max(temps) <= baseline + margin:
            break
        time.sleep(15)
    return time.monotonic() - start


def run_phase(args):
    client = Client(base_url=args.base_url, verbose=True)
    cache = PromptCache(args.prompt_cache)
    baseline_temp = max(gpu_temps() or [0])
    out = {"config": args.config, "model": client.model(),
           "baseline_temps": gpu_temps(), "sessions": []}
    system_ids = build_ids(client, SYSTEM_TOKENS, seed_of("t3sys"),
                           cache=cache, key="t3-system")
    for ctx in CONTEXTS:
        if args.config == "B" and args.reset_between_sessions:
            client.reset_prefix_cache()
            time.sleep(2)
        pad_len = max(0, ctx - SYSTEM_TOKENS - 64)
        history = system_ids + build_ids(
            client, pad_len, seed_of("t3pad", ctx),
            cache=cache, key=f"t3-pad-{ctx}") if pad_len else list(system_ids)
        session = {"target_ctx": ctx, "temps": gpu_temps(), "turns": []}
        prev_total = None
        for turn in range(1, TURNS + 1):
            user = encode_fragment(
                client,
                f"\n\n[第{turn}輪] 請用約150字延續分析,並在結尾標注輪次編號。")
            prompt = history + user
            try:
                m0 = client.metrics()
            except Exception:  # noqa: BLE001 - metrics are evidence, not gate
                m0 = None
            rec = client.complete(prompt, REPLY_TOKENS,
                                  seed=seed_of("t3s", ctx, turn))
            hit_tokens = None
            if m0 is not None:
                try:
                    m1 = client.metrics()
                    hit_tokens = int(
                        client.metric(m1, "vllm:prefix_cache_hits")
                        - client.metric(m0, "vllm:prefix_cache_hits"))
                except Exception:  # noqa: BLE001
                    hit_tokens = None
            entry = {
                "turn": turn, "prompt_len": len(prompt),
                "prev_request_total": prev_total,
                "expected_hit": (prev_total // BLOCK) * BLOCK
                if prev_total else None,
                "cached_tokens": (rec.get("cached_tokens")
                                  if rec.get("cached_tokens") is not None
                                  else hit_tokens),
                "hit_tokens_metric": hit_tokens,
                "ttft_s": rec.get("ttft_s"),
                "decode_tok_s": rec.get("decode_tok_s"),
                "ok": rec.get("ok"), "error_kind": rec.get("error_kind"),
                "completion_len": len(rec.get("token_ids") or []),
            }
            session["turns"].append(entry)
            print(f"[t3:{args.config}] ctx={ctx} turn={turn} "
                  f"prompt={entry['prompt_len']} "
                  f"ttft={entry['ttft_s'] and round(entry['ttft_s'], 2)}s "
                  f"cached={entry['cached_tokens']} "
                  f"expect~{entry['expected_hit']}", flush=True)
            if not rec.get("ok"):
                break
            history = prompt + (rec.get("token_ids") or [])
            prev_total = len(history)
        out["sessions"].append(session)
        write_json(args.out, out)
        waited = cooldown(baseline_temp, args.cooldown_seconds)
        print(f"[t3] session ctx={ctx} done; cooled {waited:.0f}s", flush=True)
    write_json(args.out, out)
    print(f"[t3:{args.config}] wrote {args.out}")


def compare_phase(args):
    b = json.load(open(args.b))
    base = json.load(open(args.baseline))
    base_by_ctx = {s["target_ctx"]: s for s in base["sessions"]}
    failures = []
    print(f"{'ctx':>8} {'turn':>4} {'B ttft':>8} {'base ttft':>9} "
          f"{'cached':>8} {'expect':>8} {'hit ok':>6}")
    for sess in b["sessions"]:
        ctx = sess["target_ctx"]
        bs = base_by_ctx.get(ctx, {"turns": []})
        for i, turn in enumerate(sess["turns"]):
            bt = bs["turns"][i] if i < len(bs["turns"]) else {}
            hit_ok = "-"
            if turn["turn"] >= 2 and turn["expected_hit"] is not None:
                cached = turn.get("cached_tokens")
                if cached is None:
                    hit_ok = "n/a"
                else:
                    ok = abs(cached - turn["expected_hit"]) <= BLOCK
                    hit_ok = "PASS" if ok else "FAIL"
                    if not ok:
                        failures.append(
                            f"ctx={ctx} turn={turn['turn']}: cached={cached} "
                            f"expected~{turn['expected_hit']} (+/-{BLOCK})")
            print(f"{ctx:>8} {turn['turn']:>4} "
                  f"{(turn.get('ttft_s') or 0):>7.2f}s "
                  f"{(bt.get('ttft_s') or 0):>8.2f}s "
                  f"{turn.get('cached_tokens') if turn.get('cached_tokens') is not None else '-':>8} "
                  f"{turn.get('expected_hit') if turn.get('expected_hit') is not None else '-':>8} "
                  f"{hit_ok:>6}")
    big = next((s for s in b["sessions"] if s["target_ctx"] == 200704), None)
    if big and len(big["turns"]) >= 2:
        t2 = big["turns"][1].get("ttft_s")
        if t2 is not None:
            mark = "PASS" if t2 < 3.0 else "FAIL"
            print(f"\n{mark}  headline: turn-2 TTFT @200k = {t2:.2f}s "
                  f"(acceptance < 3s)")
            if mark == "FAIL":
                failures.append(f"turn-2 TTFT @200k {t2:.2f}s >= 3s")
    print()
    if failures:
        print(f"T3 VERDICT: FAIL ({len(failures)} problem(s))")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("T3 VERDICT: PASS -- hit arithmetic exact, TTFT collapse delivered")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", choices=("A", "B", "C"), required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--prompt-cache", default="t3_prompt_cache.json")
    run.add_argument("--cooldown-seconds", type=float, default=120)
    run.add_argument("--reset-between-sessions", action="store_true",
                     default=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--baseline", required=True)
    args = ap.parse_args()
    run_phase(args) if args.phase == "run" else compare_phase(args)


if __name__ == "__main__":
    main()


