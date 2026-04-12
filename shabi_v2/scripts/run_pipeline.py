from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"


def run_command(command):
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full sentiment submission pipeline: scrape, prepare, train, and inference demo."
    )
    parser.add_argument(
        "--per-app-target",
        type=int,
        default=5000,
        help="Target reviews per app when scraping.",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping step and use existing raw data.",
    )
    parser.add_argument(
        "--skip-deep-learning",
        action="store_true",
        help="Skip deep learning experiment in training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exec = sys.executable

    if not args.skip_scrape:
        run_command(
            [
                python_exec,
                str(SCRIPTS_DIR / "scrape_playstore_reviews.py"),
                "--per-app-target",
                str(args.per_app_target),
            ]
        )

    run_command([python_exec, str(SCRIPTS_DIR / "prepare_dataset.py")])

    train_cmd = [python_exec, str(SCRIPTS_DIR / "train_experiments.py")]
    if args.skip_deep_learning:
        train_cmd.append("--skip-deep-learning")
    run_command(train_cmd)

    run_command([python_exec, str(SCRIPTS_DIR / "run_inference.py")])
    print("[DONE] Full pipeline completed.")


if __name__ == "__main__":
    main()
