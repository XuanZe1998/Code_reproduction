"""运行当前项目的实验并绘制论文风格结果图。"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_dqn import ExperimentConfig, SocialTrustNetwork, TDQNTrainer
from opinion_dqn.baselines import run_single_strategy
from opinion_dqn.heuristics import detect_communities


DATASET_SPECS = (
    {"name": "BBV", "kind": "synthetic"},
    {"name": "SBM", "kind": "synthetic"},
    {"name": "WS", "kind": "synthetic"},
    {
        "name": "BitcoinOTC",
        "kind": "real",
        "path": ROOT / "paper_style_results" / "data" / "soc-sign-bitcoinotc.csv",
        "url": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
    },
)
DATASET_NAMES = tuple(spec["name"] for spec in DATASET_SPECS)
METHOD_NAMES = ("T-DQN", "MaxDegree", "Blocking", "MixStrategy", "CbC", "CI")
METHOD_TO_ACTION = {
    "MaxDegree": 0,
    "Blocking": 1,
    "MixStrategy": 2,
    "CbC": 3,
    "CI": 4,
}

BASE_RANDOM_SEED = 7
POSITIVE_RATIO = 0.8
DEVICE = "cpu"
EPISODES = 2
REPEATS = 1

SEED_COUNTS = [20, 40, 60]
TIME_STEPS = [10, 20, 30, 40]
ECHO_SEED_COUNTS = [20, 40, 60]
ECHO_TIME_STEPS = [10, 20, 30, 40]

OUTPUT_DIR = ROOT / "paper_style_results"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def clear_existing_outputs() -> None:
    # 强制重跑时删除旧的实验数据和图片，避免混用历史结果。
    preserved_real_files = {
        Path(spec["path"]).resolve()
        for spec in DATASET_SPECS
        if spec["kind"] == "real"
    }
    if DATA_DIR.exists():
        for path in DATA_DIR.glob("*"):
            if path.is_file() and path.resolve() not in preserved_real_files:
                path.unlink()


def dataset_rows_complete(df: pd.DataFrame, expected_names: tuple[str, ...]) -> bool:
    if "network" not in df.columns:
        return False
    existing = set(df["network"].unique().tolist())
    return all(name in existing for name in expected_names)


def ensure_real_dataset_file(spec: dict[str, object]) -> None:
    # 若真实数据集文件不存在，则从官方地址自动下载并解压。
    path = Path(spec["path"])
    if path.exists():
        return
    url = str(spec["url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    gz_path = path.with_suffix(path.suffix + ".gz")
    print(f"downloading real dataset: {url}")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as src, path.open("wb") as dst:
        dst.write(src.read())
    gz_path.unlink(missing_ok=True)
    print(f"downloaded: {path}")
    if FIG_DIR.exists():
        for path in FIG_DIR.glob("*"):
            if path.is_file():
                path.unlink()


def make_config(run_seed: int, seed_budget: int, time_steps: int) -> ExperimentConfig:
    config = ExperimentConfig(
        seed_budget=seed_budget,
        episodes=EPISODES,
        random_seed=run_seed,
    )
    config.dynamics.time_steps = time_steps
    config.dqn.device = DEVICE
    return config


def make_random_initial_state(num_nodes: int, run_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(run_seed)
    initial_opinions = rng.uniform(-1.0, 1.0, size=num_nodes).astype(np.float32)
    initial_strategies = rng.integers(0, 2, size=num_nodes, dtype=np.int64)
    return initial_opinions, initial_strategies


def make_echo_chamber_initial_state(
    network: SocialTrustNetwork,
    run_seed: int,
    std: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(run_seed)
    community_map = detect_communities(network)
    community_ids = sorted(set(community_map.values()))
    community_centers = {cid: float(rng.uniform(-1.0, 1.0)) for cid in community_ids}

    opinions = np.zeros(network.num_nodes, dtype=np.float32)
    for node in range(network.num_nodes):
        cid = community_map[node]
        opinions[node] = np.clip(rng.normal(loc=community_centers[cid], scale=std), -1.0, 1.0)
    strategies = rng.integers(0, 2, size=network.num_nodes, dtype=np.int64)
    return opinions, strategies


def load_datasets() -> dict[str, SocialTrustNetwork]:
    datasets: dict[str, SocialTrustNetwork] = {}
    for spec in DATASET_SPECS:
        if spec["kind"] == "synthetic":
            datasets[spec["name"]] = SocialTrustNetwork.generate_paper_synthetic_network(
                spec["name"],
                seed=BASE_RANDOM_SEED,
                positive_ratio=POSITIVE_RATIO,
            )
        elif spec["kind"] == "real":
            ensure_real_dataset_file(spec)
            datasets[spec["name"]] = SocialTrustNetwork.from_snap_signed_csv(spec["path"])
        else:
            raise ValueError(f"unknown dataset kind: {spec['kind']}")
    return datasets


def run_method(
    network: SocialTrustNetwork,
    method_name: str,
    initial_opinions: np.ndarray,
    initial_strategies: np.ndarray,
    config: ExperimentConfig,
) -> float:
    if method_name == "T-DQN":
        trainer = TDQNTrainer(config)
        result = trainer.fit(network, initial_opinions, initial_strategies)
        return float(result.best_score)
    action_idx = METHOD_TO_ACTION[method_name]
    _, score = run_single_strategy(network, initial_opinions, initial_strategies, config, action_idx=action_idx)
    return float(score)


def run_grid(
    x_name: str,
    x_values: list[int],
    time_steps: int,
    seed_budget: int,
    initial_state_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    networks = load_datasets()

    for network_name, network in networks.items():
        for x_value in x_values:
            current_seed_budget = x_value if x_name == "seed_count" else seed_budget
            current_time_steps = x_value if x_name == "time_steps" else time_steps

            for repeat_idx in range(REPEATS):
                run_seed = BASE_RANDOM_SEED + repeat_idx
                if initial_state_mode == "random":
                    initial_opinions, initial_strategies = make_random_initial_state(network.num_nodes, run_seed)
                elif initial_state_mode == "echo":
                    initial_opinions, initial_strategies = make_echo_chamber_initial_state(network, run_seed)
                else:
                    raise ValueError(f"unknown initial_state_mode: {initial_state_mode}")

                for method_name in METHOD_NAMES:
                    config = make_config(run_seed, current_seed_budget, current_time_steps)
                    print(
                        f"[{x_name}] network={network_name} x={x_value} "
                        f"method={method_name} repeat={repeat_idx + 1}/{REPEATS}"
                    )
                    score = run_method(
                        network,
                        method_name,
                        initial_opinions,
                        initial_strategies,
                        config,
                    )
                    rows.append(
                        {
                            "network": network_name,
                            "x_name": x_name,
                            "x_value": x_value,
                            "method": method_name,
                            "repeat": repeat_idx,
                            "score": score,
                            "initial_state_mode": initial_state_mode,
                            "dataset_kind": next(spec["kind"] for spec in DATASET_SPECS if spec["name"] == network_name),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["network", "x_name", "x_value", "method", "initial_state_mode"], as_index=False)["score"]
        .mean()
        .sort_values(["network", "x_value", "method"])
    )


def print_console_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n===== {title} =====")
    summary = (
        df.groupby(["network", "method"], as_index=False)["score"]
        .mean()
        .sort_values(["network", "score"], ascending=[True, False])
    )
    for network_name in DATASET_NAMES:
        subset = summary[summary["network"] == network_name]
        print(f"[{network_name}]")
        for _, row in subset.iterrows():
            print(f"  {row['method']:<12} score={row['score']:.4f}")


def plot_single_network(
    df: pd.DataFrame,
    network_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    markers = {
        "T-DQN": "o",
        "MaxDegree": "s",
        "Blocking": "^",
        "MixStrategy": "D",
        "CbC": "v",
        "CI": "P",
    }
    colors = {
        "T-DQN": "#d62728",
        "MaxDegree": "#1f77b4",
        "Blocking": "#2ca02c",
        "MixStrategy": "#9467bd",
        "CbC": "#8c564b",
        "CI": "#ff7f0e",
    }
    subset = df[df["network"] == network_name]
    for method_name in METHOD_NAMES:
        method_df = subset[subset["method"] == method_name]
        ax.plot(
            method_df["x_value"],
            method_df["score"],
            marker=markers[method_name],
            color=colors[method_name],
            linewidth=2,
            markersize=6,
            label=method_name,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(force_rerun: bool = False) -> None:
    ensure_dirs()
    if force_rerun:
        clear_existing_outputs()

    seed_csv = DATA_DIR / "seed_count_results.csv"
    if seed_csv.exists():
        seed_df = pd.read_csv(seed_csv)
        if not dataset_rows_complete(seed_df, DATASET_NAMES):
            seed_df = run_grid(
                x_name="seed_count",
                x_values=SEED_COUNTS,
                time_steps=30,
                seed_budget=40,
                initial_state_mode="random",
            )
            seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")
    else:
        seed_df = run_grid(
            x_name="seed_count",
            x_values=SEED_COUNTS,
            time_steps=30,
            seed_budget=40,
            initial_state_mode="random",
        )
        seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")
    print_console_summary(seed_df, "Seed-count experiment")
    seed_summary = summarize_for_plot(seed_df)
    for network_name in DATASET_NAMES:
        plot_single_network(
            seed_summary,
            network_name=network_name,
            xlabel="Number of seed nodes",
            ylabel="Overall opinion",
            title=f"Seed count vs overall opinion ({network_name})",
            output_path=FIG_DIR / f"fig3_like_seed_count_{network_name.lower()}.png",
        )

    time_csv = DATA_DIR / "time_step_results.csv"
    if time_csv.exists():
        time_df = pd.read_csv(time_csv)
        if not dataset_rows_complete(time_df, DATASET_NAMES):
            time_df = run_grid(
                x_name="time_steps",
                x_values=TIME_STEPS,
                time_steps=30,
                seed_budget=40,
                initial_state_mode="random",
            )
            time_df.to_csv(time_csv, index=False, encoding="utf-8-sig")
    else:
        time_df = run_grid(
            x_name="time_steps",
            x_values=TIME_STEPS,
            time_steps=30,
            seed_budget=40,
            initial_state_mode="random",
        )
        time_df.to_csv(time_csv, index=False, encoding="utf-8-sig")
    print_console_summary(time_df, "Time-step experiment")
    time_summary = summarize_for_plot(time_df)
    for network_name in DATASET_NAMES:
        plot_single_network(
            time_summary,
            network_name=network_name,
            xlabel="Time steps",
            ylabel="Overall opinion",
            title=f"Time steps vs overall opinion ({network_name})",
            output_path=FIG_DIR / f"fig4_like_time_steps_{network_name.lower()}.png",
        )

    echo_csv = DATA_DIR / "echo_sbm_results.csv"
    if echo_csv.exists():
        echo_df = pd.read_csv(echo_csv)
        echo_seed_df = echo_df[(echo_df["network"] == "SBM") & (echo_df["x_name"] == "seed_count")].copy()
        echo_time_df = echo_df[(echo_df["network"] == "SBM") & (echo_df["x_name"] == "time_steps")].copy()
    else:
        echo_seed_df = run_grid(
            x_name="seed_count",
            x_values=ECHO_SEED_COUNTS,
            time_steps=30,
            seed_budget=40,
            initial_state_mode="echo",
        )
        echo_time_df = run_grid(
            x_name="time_steps",
            x_values=ECHO_TIME_STEPS,
            time_steps=30,
            seed_budget=40,
            initial_state_mode="echo",
        )
        echo_df = pd.concat(
            [echo_seed_df[echo_seed_df["network"] == "SBM"], echo_time_df[echo_time_df["network"] == "SBM"]],
            ignore_index=True,
        )
        echo_df.to_csv(echo_csv, index=False, encoding="utf-8-sig")
    print_console_summary(pd.concat([echo_seed_df, echo_time_df], ignore_index=True), "Echo-chamber SBM experiment")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    markers = {
        "T-DQN": "o",
        "MaxDegree": "s",
        "Blocking": "^",
        "MixStrategy": "D",
        "CbC": "v",
        "CI": "P",
    }
    colors = {
        "T-DQN": "#d62728",
        "MaxDegree": "#1f77b4",
        "Blocking": "#2ca02c",
        "MixStrategy": "#9467bd",
        "CbC": "#8c564b",
        "CI": "#ff7f0e",
    }

    echo_seed_summary = summarize_for_plot(echo_seed_df[echo_seed_df["network"] == "SBM"])
    echo_time_summary = summarize_for_plot(echo_time_df[echo_time_df["network"] == "SBM"])

    for method_name in METHOD_NAMES:
        seed_method_df = echo_seed_summary[echo_seed_summary["method"] == method_name]
        axes[0].plot(
            seed_method_df["x_value"],
            seed_method_df["score"],
            marker=markers[method_name],
            color=colors[method_name],
            linewidth=2,
            markersize=6,
            label=method_name,
        )
        time_method_df = echo_time_summary[echo_time_summary["method"] == method_name]
        axes[1].plot(
            time_method_df["x_value"],
            time_method_df["score"],
            marker=markers[method_name],
            color=colors[method_name],
            linewidth=2,
            markersize=6,
            label=method_name,
        )

    axes[0].set_title("Echo chamber on SBM: seed count")
    axes[0].set_xlabel("Number of seed nodes")
    axes[0].set_ylabel("Overall opinion")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("Echo chamber on SBM: time steps")
    axes[1].set_xlabel("Time steps")
    axes[1].set_ylabel("Overall opinion")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "fig5_like_echo_sbm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    config_summary = {
        "device": DEVICE,
        "episodes": EPISODES,
        "repeats": REPEATS,
        "datasets": list(DATASET_NAMES),
        "seed_counts": SEED_COUNTS,
        "time_steps": TIME_STEPS,
        "echo_seed_counts": ECHO_SEED_COUNTS,
        "echo_time_steps": ECHO_TIME_STEPS,
        "methods": list(METHOD_NAMES),
    }
    (DATA_DIR / "plot_config.json").write_text(json.dumps(config_summary, indent=2), encoding="utf-8")
    print(f"figures written to: {FIG_DIR}")
    print(f"raw csv written to: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments and generate paper-style figures.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="delete old csv/png files and rerun all experiments from scratch",
    )
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
