"""Select one of the five policies and collect all runs through one command."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from tools import tutorial_rl as archive

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "experiments/tutorial_suite/models"
SPECIALISTS = {
    2: ("stage_02.zip", "2357750478d0433fa903cfc76387481eb0f69460d42e46a083cf4c71f1a5283a"),
    4: ("stage_04.pt", "87e5fc82772acd185d7428e1fcabed7aa0c22aa8499f626abcb22b1d68f825d0"),
}


def policy(stage):
    if stage in SPECIALISTS:
        name, digest = SPECIALISTS[stage]
        return ROOT, MODELS / name, digest, f"stage-{stage:02d}-specialist"
    entry, = (model for model in archive.models() if
              (model["stage"], model["experiment"], model["algorithm"], model["training_seed"])
              == (stage, "baseline", "ppo", 0))
    archive.check_model(entry)
    return archive.ARCHIVE / entry["runtime"], archive.ARCHIVE / entry["checkpoint"], entry["sha256"], entry["id"]


def run_stage(stage, args, result_path):
    runtime, model, digest, label = policy(stage)
    cmd = [sys.executable, "-B", str(Path(__file__).with_name("worker.py")),
           "--mode", args.command, "--stage", str(stage), "--runtime", str(runtime), "--seed", str(args.seed),
           "--output", str(result_path)]
    if args.command == "train":
        cmd += ["--steps", str(args.steps)]
    else:
        override = getattr(args, "checkpoint", None)
        if override:
            model, digest, label = override.resolve(), None, override.name
        cmd += ["--checkpoint", str(model), "--model", label, "--episodes", str(args.episodes)]
        if digest:
            cmd += ["--sha256", digest]
        if args.command == "verify" or getattr(args, "greedy", False):
            cmd += ["--greedy"]
        if getattr(args, "headless", False):
            cmd += ["--headless"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    if args.command != "train" and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stages = list(map(str, range(1, 6)))
    for action in ("list", "verify", "evaluate", "play", "train"):
        command = sub.add_parser(action)
        command.add_argument("--suite", choices=("combined",), default="combined")
        if action == "list":
            continue
        command.add_argument("--seed", type=int, default=120000)
        if action in ("verify", "evaluate"):
            command.add_argument("--stage", choices=[*stages, "all"], default="all")
        else:
            command.add_argument("--stage", choices=stages, required=True)
        if action == "train":
            command.add_argument("--steps", type=int, required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--output", type=Path)
        if action == "evaluate":
            command.add_argument("--episodes", type=int, default=100)
        else:
            command.set_defaults(episodes=1)
        if action in ("evaluate", "play"):
            command.add_argument("--checkpoint", type=Path, help="Trusted SB3 .zip from the combined training command")
        if action == "play":
            command.add_argument("--greedy", action="store_true")
            command.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        for stage in range(1, 6):
            _, checkpoint, _, label = policy(stage)
            print(f"{stage}: {label} ({checkpoint.relative_to(ROOT).as_posix()})")
        return
    if args.seed < 0 or args.episodes <= 0 or (args.command == "train" and args.steps <= 0):
        parser.error("seed must be nonnegative and steps/episodes must be positive")
    if getattr(args, "checkpoint", None) and (args.stage == "all" or args.checkpoint.suffix.lower() != ".zip"):
        parser.error("--checkpoint requires a single stage and an SB3 .zip model")
    if args.output and args.output.exists():
        parser.error(f"Choose a new output path: {args.output}")
    if args.command == "train":
        run_stage(int(args.stage), args, args.output.resolve())
        return

    selection = list(range(1, 6)) if args.stage == "all" else [int(args.stage)]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        with ThreadPoolExecutor(max_workers=min(3, len(selection))) as executor:
            futures = {stage: executor.submit(run_stage, stage, args, Path(temporary) / f"{stage}.json")
                       for stage in selection}
            for stage, future in futures.items():
                row = future.result()
                if row is None:
                    continue
                rows.append(row)
                print(f"Stage {stage}: {row['successes']}/{row['episodes']}, "
                      f"mean score {row['mean_score']:.2f}", flush=True)
    if args.output and rows:
        protocol = "greedy" if args.command == "verify" or getattr(args, "greedy", False) else "categorical_sampling"
        write_report(args.output, dict(suite="combined", command=args.command, protocol=protocol,
                                       seed=args.seed, stages=rows))
