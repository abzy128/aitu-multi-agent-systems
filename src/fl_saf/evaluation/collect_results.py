from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment metric CSVs")
    parser.add_argument("--root", default="output")
    parser.add_argument("--prefix", default="track_a_seed0_")
    parser.add_argument("--out", default="output/track_a_seed0_summary.csv")
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


if __name__ == "__main__":
    main()
