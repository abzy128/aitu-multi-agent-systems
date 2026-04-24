from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment metric CSVs")
    parser.add_argument("--root", default="output")
    parser.add_argument("--prefix", default="track_a_seed")
    parser.add_argument("--out", default="output/track_a_summary.csv")
    parser.add_argument("--stats-out", default="output/track_a_summary_stats.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.glob(f"{args.prefix}*/metrics.csv")):
        run_id = metrics_path.parent.name
        with metrics_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append({"run_id": run_id, **row})
    if not rows:
        raise SystemExit(f"no metrics found under {root} with prefix {args.prefix!r}")
    keys = sorted({key for row in rows for key in row})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    stats_rows = summarize(rows)
    stats_keys = sorted({key for row in stats_rows for key in row})
    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with stats_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=stats_keys)
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"wrote {len(stats_rows)} rows to {stats_out}")


def summarize(rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["algorithm"], row["client"]), []).append(row)

    metric_names = ["rmse", "mae", "mape", "r2", "comm_mb", "comm_rounds"]
    out = []
    for (algorithm, client), group_rows in sorted(groups.items()):
        summary: dict[str, float | str | int] = {
            "algorithm": algorithm,
            "client": client,
            "n_seeds": len(group_rows),
        }
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric) not in (None, "")
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = math.sqrt(variance)
        out.append(summary)
    return out


if __name__ == "__main__":
    main()
