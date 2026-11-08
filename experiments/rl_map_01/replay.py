"""Verify the preserved files and replay one policy on its own source snapshot."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ARCHIVE = Path(__file__).resolve().parent
RUNS = ("original_start", "curriculum_start")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root, hashes):
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"File missing or SHA-256 mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=RUNS, nargs="?")
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    args = parser.parse_args()
    hashes = read(ARCHIVE / "sha256.json")
    verify(ARCHIVE, hashes)
    print(f"Archive integrity verified: {len(hashes)} files.")
    if args.run is None:
        return

    run = ARCHIVE / args.run
    snapshot = run / "source_snapshot"
    data = run / "seed_2026"
    verify(snapshot, read(data / "config.json")["sources"])
    sys.path.insert(0, str(snapshot))

    import torch
    from rl.map1_spec_train import evaluate
    from rl.map_a_models import load_policy

    torch.set_num_threads(1)
    model = load_policy(data / f"{args.checkpoint}.zip")
    summary = read(data / "summary.json")
    if args.checkpoint == "best":
        expected, expected_step = summary["selected_greedy"], summary["best_step"]
    else:
        expected, expected_step = summary["latest_greedy"], summary["training_steps"]
    if model.num_timesteps != expected_step:
        raise SystemExit(f"Unexpected model step: {model.num_timesteps} (expected {expected_step})")

    actual = evaluate(model, count=1, seed=expected["seed"], greedy=True, native=True)
    fields = ("successes", "mean_return", "mean_reward_items", "mean_success_seconds",
              "mean_episode_seconds", "max_native_error")
    for field in fields:
        a, e = actual[field], expected[field]
        if a is None or e is None:
            matches = a is e
        else:
            matches = math.isclose(a, e, rel_tol=0.0, abs_tol=1e-8)
        if not matches:
            raise SystemExit(f"Replay mismatch in {field}: {a} (expected {e})")
    print(json.dumps(dict(run=args.run, checkpoint=args.checkpoint, step=model.num_timesteps,
                          **{field: actual[field] for field in fields}), indent=2))


if __name__ == "__main__":
    main()
