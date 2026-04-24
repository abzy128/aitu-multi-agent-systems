from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALGO_ORDER = ["centralized", "local_only", "fedavg", "fedprox", "scaffold", "fedbn"]
ALGO_LABELS = {
    "centralized": "Centralized",
    "local_only": "Local only",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "fedbn": "FedBN",
}
CLIENT_LABELS = {"client_1": "1", "client_2": "2"}


def fmt(mean: str, std: str, digits: int) -> str:
    if mean in (None, ""):
        return "--"
    m = float(mean)
    s = float(std) if std not in (None, "") else 0.0
    return f"${m:.{digits}f} \\pm {s:.{digits}f}$"


def fmt_int(value: str) -> str:
    if value in (None, ""):
        return "--"
    return f"{int(round(float(value)))}"


def build_table(rows: list[dict[str, str]], caption: str, label: str) -> str:
    by_key = {(r["algorithm"], r["client"]): r for r in rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Algorithm & Client & RMSE (MW) & MAE (MW) & $R^2$ & Rounds & Comm.\ (MB) \\",
        r"\midrule",
    ]
    for algo in ALGO_ORDER:
        for client in ("client_1", "client_2"):
            row = by_key.get((algo, client))
            if row is None:
                continue
            line = " & ".join([
                ALGO_LABELS[algo],
                CLIENT_LABELS[client],
                fmt(row.get("rmse_mean", ""), row.get("rmse_std", ""), 3),
                fmt(row.get("mae_mean", ""), row.get("mae_std", ""), 3),
                fmt(row.get("r2_mean", ""), row.get("r2_std", ""), 3),
                fmt_int(row.get("comm_rounds_mean", "")),
                fmt(row.get("comm_mb_mean", ""), row.get("comm_mb_std", ""), 2),
            ])
            lines.append(line + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def read_stats(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render summary_stats CSV as a LaTeX table")
    parser.add_argument("--stats", default="output/track_a_summary_stats.csv")
    parser.add_argument("--out", default="output/track_a_table.tex")
    parser.add_argument("--caption", default="Track A univariate active-power forecasting. Mean $\\pm$ std over 5 seeds.")
    parser.add_argument("--label", default="tab:track_a")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_stats(Path(args.stats))
    table = build_table(rows, args.caption, args.label)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
