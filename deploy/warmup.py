#!/usr/bin/env python3
"""Walk the prefill chunk-count ladder over HTTP, once, after boot.

The engine's own warmup (``VLLM_DSV4_WARMUP``) covers the compressed-gather
chunk ladder and the mixed token sizes, but not the axis Triton actually
buckets on: how many prefill chunks one request consumes. That axis is tied to
request length, so the first request at every unseen length still pays 7-11.5
seconds of compilation and runs at roughly half speed. Left un-warmed it looks
exactly like a broken deployment -- a long prefill appears to hang, and every
number you measure is about half of what the machine can do.

Run this once after the server reports ready, before putting traffic on it or
measuring anything. Stdlib only, like everything in verify/.

    python3 warmup.py --base-url http://127.0.0.1:8000

The default ladder stops at 180k on purpose. Warming is eight long prefills
back to back in one process, and with expandable_segments the allocator only
grows -- measured, a 240k prefill at the end of that run OOMs on a box that
serves 240k fine from a fresh boot. If you serve longer than 180k, add the
length with --lengths and read the "OOM hang when sweeping at 190k and above"
section in docs/troubleshooting.md first. There is no point compiling a bucket
you will never serve, either: trim the ladder if your deployment caps lower.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

# Roughly one bucket per step. Each number below tokenizes to about three
# tokens, so the prompt builder targets tokens, not words. See the module
# docstring for why this stops at 180k rather than at max_model_len.
DEFAULT_LENGTHS = [6000, 18000, 36000, 66000, 102000, 135000, 180000]


def build_prompt(target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    # ~3 tokens per 4-digit number plus the space.
    count = max(1, target_tokens // 3)
    return " ".join(str(rng.randrange(1000, 10000)) for _ in range(count))


def post(base_url: str, payload: dict, timeout: float) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read())
    return time.time() - started, parsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv4-flash-0731")
    ap.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="token targets to warm; default walks up to near 262144",
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--json-out", help="write the per-length results here")
    args = ap.parse_args()

    results = []
    print(f"warming {args.base_url}", flush=True)
    print(f"{'tokens':>9}  {'seconds':>8}  {'tok/s':>7}", flush=True)

    started = time.time()
    for i, target in enumerate(args.lengths):
        prompt = build_prompt(target, seed=1000 + i)
        try:
            # max_tokens=1: this is about compiling the prefill path, not decode.
            elapsed, parsed = post(
                args.base_url,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                    "ignore_eos": True,
                },
                args.timeout,
            )
        except urllib.error.HTTPError as exc:
            results.append({"target": target, "error": f"HTTP {exc.code}"})
            if exc.code >= 500:
                # The engine died, most likely OOM on a long prefill. Every
                # later length would fail the same way, so stop rather than
                # keep hammering a dead server.
                print(f"{target:>9}  HTTP {exc.code} -- the engine is probably "
                      f"down; stopping", flush=True)
                print("  check the log for OutOfMemoryError, and see the "
                      "'OOM hang when sweeping at 190k and above' section in "
                      "docs/troubleshooting.md", flush=True)
                break
            # 400 usually means the target overshot max_model_len; that bucket
            # is not servable anyway, so it is not worth compiling.
            print(f"{target:>9}  skipped ({exc.code})", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"{target:>9}  FAILED {type(exc).__name__}: {exc}", flush=True)
            results.append({"target": target, "error": repr(exc)})
            continue

        tokens = parsed["usage"]["prompt_tokens"]
        print(f"{tokens:>9}  {elapsed:>8.1f}  {tokens / elapsed:>7.0f}", flush=True)
        results.append(
            {"target": target, "prompt_tokens": tokens, "seconds": elapsed,
             "prefill_tokens_per_s": tokens / elapsed}
        )

    # One short decode so the captured graph replays at least once before real
    # traffic; the capture itself happens at boot, this just exercises it.
    try:
        post(
            args.base_url,
            {"model": args.model, "prompt": "1 2 3 4 5", "max_tokens": 64,
             "temperature": 0, "ignore_eos": True},
            300.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"decode warmup failed: {type(exc).__name__}: {exc}", flush=True)

    total = time.time() - started
    failed = [r for r in results if "error" in r]
    print(f"\ndone in {total:.0f}s; {len(results) - len(failed)}/{len(results)} lengths warmed")
    if failed:
        print("the failed lengths will pay the compile tax on their first real request")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"base_url": args.base_url, "seconds": total,
                       "results": results}, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
