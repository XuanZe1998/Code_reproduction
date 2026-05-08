"""生成论文中的三类人工合成网络并按 SNAP 格式导出到本地。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # 让脚本在仓库根目录直接执行时也能找到源码包。
    sys.path.insert(0, str(SRC))

from opinion_dqn.graph import SocialTrustNetwork


def export_network(name: str, network: SocialTrustNetwork, out_dir: Path) -> dict[str, float]:
    # 每个网络导出一份 SNAP 风格边表，同时返回摘要统计值。
    out_path = out_dir / f"soc-sign-{name.lower()}.csv"
    network.export_snap_signed_csv(out_path, include_header=False)
    summary = network.summary()
    print(
        f"{name}: nodes={int(summary['nodes'])}, "
        f"edges={int(summary['edges'])}, "
        f"avg_degree={summary['avg_degree']:.4f}, "
        f"positive_ratio={summary['positive_ratio']:.4f}"
    )
    return summary


def main() -> None:
    # 按论文使用的合成网络类型逐个生成并导出。
    out_dir = ROOT / "generated_networks"
    out_dir.mkdir(exist_ok=True)

    summaries: dict[str, dict[str, float]] = {}
    for name in ("BBV", "SBM", "WS"):
        network = SocialTrustNetwork.generate_paper_synthetic_network(name, seed=7, positive_ratio=0.8)
        summaries[name] = export_network(name, network, out_dir)

    summary_path = out_dir / "network_summaries.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"written: {summary_path}")


if __name__ == "__main__":
    main()
