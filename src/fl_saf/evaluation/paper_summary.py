from __future__ import annotations

import argparse
import csv
from pathlib import Path


MAIN_15_ROUND = {"centralized", "local_only", "fedavg", "fedbn"}
MAIN_50_ROUND = {"fedprox", "scaffold"}
ALGO_LABELS = {
    "centralized": "Centralized",
    "local_only": "Local only",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "fedbn": "FedBN",
}
CLIENT_LABELS = {"client_1": "Client 1", "client_2": "Client 2"}
ALGO_ORDER = {
    "centralized": 0,
    "local_only": 1,
    "fedavg": 2,
    "fedprox": 3,
    "scaffold": 4,
    "fedbn": 5,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(row: dict[str, str], metric: str, digits: int = 3) -> str:
    mean = row.get(f"{metric}_mean", "")
    std = row.get(f"{metric}_std", "")
    if mean in ("", None):
        return "--"
    return f"{float(mean):.{digits}f} +/- {float(std or 0.0):.{digits}f}"


def with_source(rows: list[dict[str, str]], source: str) -> list[dict[str, str]]:
    return [{**row, "source": source} for row in rows]


def select_recommended(
    track_a_rows: list[dict[str, str]], r50_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    recommended = []
    for row in track_a_rows:
        if row["algorithm"] in MAIN_15_ROUND:
            recommended.append({**row, "paper_round_choice": "15"})
    for row in r50_rows:
        if row["algorithm"] in MAIN_50_ROUND:
            recommended.append({**row, "paper_round_choice": "50"})
    return sorted(recommended, key=lambda r: (ALGO_ORDER[r["algorithm"]], r["client"]))


def build_markdown(
    recommended: list[dict[str, str]],
    track_a_rows: list[dict[str, str]],
    r50_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Track A Research Summary",
        "",
        "## Recommended Main Table",
        "",
        "Use the 15-round table for centralized, local-only, FedAvg, and FedBN. "
        "Use the 50-round ablation values for FedProx and SCAFFOLD, because both "
        "methods materially improve with more rounds in this setup.",
        "",
        "| Algorithm | Client | Rounds | RMSE (MW) | MAE (MW) | R2 | Comm. MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in recommended:
        rounds = row.get("comm_rounds_mean", "")
        rounds_label = str(int(round(float(rounds)))) if rounds else "--"
        comm = fmt(row, "comm_mb", 2) if row.get("comm_mb_mean") else "--"
        lines.append(
            " | ".join(
                [
                    f"| {ALGO_LABELS[row['algorithm']]}",
                    CLIENT_LABELS[row["client"]],
                    rounds_label,
                    fmt(row, "rmse"),
                    fmt(row, "mae"),
                    fmt(row, "r2"),
                    comm + " |",
                ]
            )
        )

    lines += [
        "",
        "## Paper Claims Supported By Current Outputs",
        "",
        "- FedProx is the strongest FL method on the primary metric when allowed 50 rounds: "
        "it reaches the best client-1 RMSE and remains competitive on client 2.",
        "- SCAFFOLD no longer diverges after the optimizer fix, but it needs more rounds "
        "and remains communication-heavy.",
        "- FedBN is not a clear win for this two-client Track A setting. The corrected "
        "implementation is acceptable at 15 rounds but degrades badly at 50 rounds, "
        "which is useful evidence about its limitation under this client mix.",
        "- FedAvg is a reasonable 15-round baseline, but the 50-round ablation shows "
        "client-2 degradation rather than monotonic improvement.",
        "",
        "## Round Ablation",
        "",
        "| Algorithm | Client | 15-round RMSE | 50-round RMSE | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    by_15 = {(r["algorithm"], r["client"]): r for r in track_a_rows}
    by_50 = {(r["algorithm"], r["client"]): r for r in r50_rows}
    for algorithm in ("fedavg", "fedprox", "scaffold", "fedbn"):
        for client in ("client_1", "client_2"):
            r15 = by_15[(algorithm, client)]
            r50 = by_50[(algorithm, client)]
            rmse15 = float(r15["rmse_mean"])
            rmse50 = float(r50["rmse_mean"])
            direction = "improves" if rmse50 < rmse15 else "worsens"
            lines.append(
                f"| {ALGO_LABELS[algorithm]} | {CLIENT_LABELS[client]} | "
                f"{fmt(r15, 'rmse')} | {fmt(r50, 'rmse')} | {direction} |"
            )

    lines += [
        "",
        "## Files",
        "",
        "- `output/track_a_summary_stats.csv`: 15-round Track A summary over five seeds.",
        "- `output/track_a_r50_summary_stats.csv`: 50-round FL ablation over five seeds.",
        "- `output/research/track_a_recommended_summary_stats.csv`: selected paper-facing table.",
        "- `output/research/track_a_round_ablation_summary_stats.csv`: combined 15/50 FL rows.",
        "- `output/figures/`: validation-loss, prediction-overlay, and Pareto figures.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing Track A result files")
    parser.add_argument("--track-a", default="output/track_a_summary_stats.csv")
    parser.add_argument("--r50", default="output/track_a_r50_summary_stats.csv")
    parser.add_argument("--out-dir", default="output/research")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track_a_rows = read_rows(Path(args.track_a))
    r50_rows = read_rows(Path(args.r50))
    out_dir = Path(args.out_dir)

    recommended = select_recommended(track_a_rows, r50_rows)
    ablation = with_source(
        [r for r in track_a_rows if r["algorithm"] not in {"centralized", "local_only"}],
        "15_round",
    ) + with_source(r50_rows, "50_round")

    write_rows(out_dir / "track_a_recommended_summary_stats.csv", recommended)
    write_rows(out_dir / "track_a_round_ablation_summary_stats.csv", ablation)
    (out_dir / "track_a_research_summary.md").write_text(
        build_markdown(recommended, track_a_rows, r50_rows)
    )
    print(f"wrote paper-facing summaries to {out_dir}")


if __name__ == "__main__":
    main()
