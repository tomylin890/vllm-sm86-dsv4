#!/usr/bin/env python3
"""T4 -- eviction pressure: does the B4 pathology crush long-prefix hits?

  python3 p11_t4_eviction.py run --log <serve-B-*.log> --out t4.json

REQUIRES a boot with a larger pool and VLLM_LOGGING_LEVEL=DEBUG (the
"Cache hit reconciliation" line is debug-gated). 4 concurrent agent sessions,
each >= 100k context, one turn per session per round, >= 8 rounds (>= 32
turns). After every round: parse the new reconciliation lines from the serve
log and apply the thermal gate (cooldown + temps back to baseline + 5 C).

B4 pathology signature (design T4): an MLA (FullAttention) group reports a
first-sighting hit near the full prefix while a sliding-window/state group
reports far less -- the MIN reconciliation then crushes the whole hit. Every
event where reconciled < 0.5 * longest per-group is flagged, with the groups
named by the `shrunk by` field.
"""

import argparse
import ast
import concurrent.futures as cf
import json
import re
import subprocess
import time

from p11lib import Client, PromptCache, build_ids, encode_fragment, seed_of, write_json

SESSIONS = 4
ROUNDS = 8
SESSION_CTX = 102400
REPLY_TOKENS = 200
RECON_RE = re.compile(
    r"Cache hit reconciliation: reconciled=(\d+), requested=(\d+), "
    r"longest per-group=(\d+), per group at first sighting "
    r"\(spec, group_ids, hit_length\)=(\[.*?\]), shrunk by "
    r"\(spec, group_ids, from, to\)=(\[.*\])")


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


class LogFollower:
    def __init__(self, path):
        self.path, self.pos = path, 0
        try:
            import os
            self.pos = os.path.getsize(path)  # only NEW lines count
        except OSError:
            pass

    def new_events(self):
        events = []
        try:
            with open(self.path, errors="replace") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
                self.pos = fh.tell()
        except OSError:
            return events
        for m in RECON_RE.finditer(chunk):
            try:
                groups = ast.literal_eval(m.group(4))
                shrunk = ast.literal_eval(m.group(5))
            except (ValueError, SyntaxError):
                groups, shrunk = [], []
            events.append({
                "reconciled": int(m.group(1)),
                "requested": int(m.group(2)),
                "longest": int(m.group(3)),
                "per_group_first_sighting": groups,
                "shrunk_by": shrunk,
            })
        return events


def session_turn(client, state, sess_id, rnd):
    user = encode_fragment(
        client, f"\n\n[會話{sess_id} 第{rnd}輪] 請延續分析並標注輪次。")
    prompt = state["history"] + user
    rec = client.complete(prompt, REPLY_TOKENS,
                          seed=seed_of("t4", sess_id, rnd))
    if rec.get("ok"):
        state["history"] = prompt + (rec.get("token_ids") or [])
    return {
        "session": sess_id, "round": rnd, "prompt_len": len(prompt),
        "ok": rec.get("ok"), "error_kind": rec.get("error_kind"),
        "ttft_s": rec.get("ttft_s"), "cached_tokens": rec.get("cached_tokens"),
        "decode_tok_s": rec.get("decode_tok_s"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", choices=("run",))
    ap.add_argument("--log", required=True, help="serve-B-*.log (DEBUG boot)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompt-cache", default="t4_prompt_cache.json")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--cooldown-seconds", type=float, default=180)
    args = ap.parse_args()

    client = Client(base_url=args.base_url, verbose=True)
    cache = PromptCache(args.prompt_cache)
    follower = LogFollower(args.log)
    if not follower.pos and not follower.new_events():
        print(f"WARN: {args.log} empty or unreadable -- is this the DEBUG "
              "boot? Reconciliation evidence will be missing.")
    baseline_temp = max(gpu_temps() or [0])
    states = {
        s: {"history": build_ids(client, SESSION_CTX,
                                 seed_of("t4hist", s), cache=cache,
                                 key=f"t4-{s}")}
        for s in range(SESSIONS)
    }
    out = {"turns": [], "recon_events": [], "pathology": [],
           "thermal": [], "baseline_temps": gpu_temps()}
    for rnd in range(1, args.rounds + 1):
        started = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=SESSIONS) as pool:
            futs = [pool.submit(session_turn, client, states[s], s, rnd)
                    for s in range(SESSIONS)]
            for f in cf.as_completed(futs):
                turn = f.result()
                out["turns"].append(turn)
                print(f"[t4] s{turn['session']} r{turn['round']} "
                      f"prompt={turn['prompt_len']} "
                      f"ttft={turn['ttft_s'] and round(turn['ttft_s'], 2)}s "
                      f"cached={turn['cached_tokens']} ok={turn['ok']}",
                      flush=True)
        events = follower.new_events()
        for ev in events:
            if ev["longest"] > 0 and ev["reconciled"] < 0.5 * ev["longest"]:
                ev_flag = {**ev, "round": rnd}
                out["pathology"].append(ev_flag)
                print(f"[t4] B4 PATHOLOGY: reconciled={ev['reconciled']} << "
                      f"longest={ev['longest']}; shrunk_by={ev['shrunk_by']}",
                      flush=True)
        out["recon_events"].extend({**e, "round": rnd} for e in events)
        write_json(args.out, out)
        load_s = time.monotonic() - started
        t0 = time.monotonic()
        time.sleep(args.cooldown_seconds)
        for _ in range(60):
            temps = gpu_temps()
            if not temps or max(temps) <= baseline_temp + 5:
                break
            time.sleep(15)
        out["thermal"].append({"round": rnd, "load_s": load_s,
                               "cooled_s": time.monotonic() - t0,
                               "temps": gpu_temps()})
        write_json(args.out, out)
    ok_turns = [t for t in out["turns"] if t["ok"]]
    warm = [t for t in ok_turns if t["round"] > 1 and t["cached_tokens"]]
    print(f"\nT4 SUMMARY: {len(ok_turns)}/{len(out['turns'])} turns ok, "
          f"{len(out['recon_events'])} reconciliation events, "
          f"{len(out['pathology'])} pathology flags, "
          f"{len(warm)} warm turns with nonzero hit")
    if out["pathology"]:
        print("T4 VERDICT: B4 PATHOLOGY OBSERVED -- P12 input, see JSON")
    else:
        print("T4 VERDICT: no hit-crush pathology under pressure")


if __name__ == "__main__":
    main()


