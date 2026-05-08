"""统一入口：生成网络、运行实验、控制台打印结果并生成图片。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str], title: str) -> None:
    print(f"\n===== {title} =====")
    print("command:", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(force_rerun: bool = False) -> None:
    python_exe = sys.executable

    run_step(
        [python_exe, "examples/generate_synthetic_networks.py"],
        "Step 1/2: generate synthetic networks",
    )

    plot_command = [python_exe, "examples/plot_paper_style_results.py"]
    if force_rerun:
        plot_command.append("--force-rerun")
    run_step(
        plot_command,
        "Step 2/2: run experiments and generate figures",
    )

    print("\n===== Done =====")
    print(f"figures: {ROOT / 'paper_style_results' / 'figures'}")
    print(f"csv: {ROOT / 'paper_style_results' / 'data'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full project pipeline and generate figures.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="rerun all experiments from scratch instead of reusing existing csv files",
    )
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
