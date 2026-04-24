"""Exploratory analysis of the Furnace1 / Furnace2 datasets.

Generates summary statistics, feature-engineered views, and a set of charts
saved under dataset/figures/ that feed into dataset/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_DIR = HERE / "dataset"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

DATASETS = {
    "Furnace1": DATA_DIR / "Furnace1.csv",
    "Furnace2": DATA_DIR / "Furnace2.csv",
}


def load(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def summarise(name: str, df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": int(numeric.shape[1]),
        "start": str(df["DateTime"].min()),
        "end": str(df["DateTime"].max()),
        "duration_hours": round(
            (df["DateTime"].max() - df["DateTime"].min()).total_seconds() / 3600, 2
        ),
        "sampling_seconds": int(
            df["DateTime"].diff().dt.total_seconds().median()
        ),
        "missing_values": int(df.isna().sum().sum()),
        "constant_columns": [c for c in numeric.columns if numeric[c].nunique() <= 1],
        "features": list(df.columns),
    }


def plot_missing(name: str, df: pd.DataFrame) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    if miss.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    miss[miss > 0].plot.bar(ax=ax, color="#d9534f")
    ax.set_title(f"{name} — Missing value ratio")
    ax.set_ylabel("ratio")
    fig.savefig(FIG_DIR / f"{name}_missing.png")
    plt.close(fig)


def plot_distributions(name: str, df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    cols = [c for c in numeric.columns if numeric[c].nunique() > 1]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(numeric[col].dropna(), bins=40, ax=ax, color="#337ab7")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{name} — Feature distributions", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_distributions.png")
    plt.close(fig)


def plot_correlation(name: str, df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title(f"{name} — Correlation matrix")
    fig.savefig(FIG_DIR / f"{name}_correlation.png")
    plt.close(fig)


def plot_timeseries(name: str, df: pd.DataFrame, cols: list[str]) -> None:
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 1.8 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df["DateTime"], df[col], color="#2c3e50", linewidth=0.6)
        ax.set_ylabel(col, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"{name} — Key signals over time", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_timeseries.png")
    plt.close(fig)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a few physically meaningful features used by the charts."""
    out = df.copy()
    out["hour"] = out["DateTime"].dt.hour
    out["minute_of_day"] = out["DateTime"].dt.hour * 60 + out["DateTime"].dt.minute

    volt_cols = [c for c in out.columns if c.startswith("voltage_") and c[-1] in "abc"]
    curr_cols = [c for c in out.columns if c.startswith("current_") and c[-1] in "abc"]
    if len(volt_cols) == 3:
        out["voltage_mean"] = out[volt_cols].mean(axis=1)
        out["voltage_imbalance"] = out[volt_cols].std(axis=1)
    if len(curr_cols) == 3:
        out["current_mean"] = out[curr_cols].mean(axis=1)
        out["current_imbalance"] = out[curr_cols].std(axis=1)
    if "active_power" in out.columns and "reactive_power" in out.columns:
        out["apparent_power"] = np.sqrt(
            out["active_power"] ** 2 + out["reactive_power"] ** 2
        )
        denom = out["apparent_power"].replace(0, np.nan)
        out["derived_cos_phi"] = out["active_power"] / denom
    out["active_power_roll5"] = (
        out["active_power"].rolling(5, min_periods=1).mean()
        if "active_power" in out.columns
        else np.nan
    )
    return out


def plot_engineered(name: str, df: pd.DataFrame) -> list[str]:
    created: list[str] = []
    engineered_cols = [
        "voltage_mean",
        "voltage_imbalance",
        "current_mean",
        "current_imbalance",
        "apparent_power",
        "derived_cos_phi",
    ]
    present = [c for c in engineered_cols if c in df.columns and df[c].nunique() > 1]
    if not present:
        return created

    fig, axes = plt.subplots(len(present), 1, figsize=(11, 1.6 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, col in zip(axes, present):
        ax.plot(df["DateTime"], df[col], linewidth=0.6, color="#8e44ad")
        ax.set_ylabel(col, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"{name} — Engineered features over time", y=1.01, fontsize=13)
    fig.tight_layout()
    fname = f"{name}_engineered.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    created.append(fname)
    return created


def plot_pca(name: str, df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 1].dropna()
    if numeric.shape[1] < 2:
        return
    scaled = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=min(6, scaled.shape[1]))
    pca.fit(scaled)
    fig, ax = plt.subplots(figsize=(6, 4))
    comps = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    ax.bar(comps, pca.explained_variance_ratio_, color="#17a2b8")
    ax.plot(comps, np.cumsum(pca.explained_variance_ratio_), color="#d35400", marker="o")
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(f"{name} — PCA scree")
    fig.savefig(FIG_DIR / f"{name}_pca.png")
    plt.close(fig)


def plot_feature_overlap(frames: dict[str, pd.DataFrame]) -> dict:
    f1 = set(frames["Furnace1"].columns)
    f2 = set(frames["Furnace2"].columns)
    shared = sorted(f1 & f2)
    only1 = sorted(f1 - f2)
    only2 = sorted(f2 - f1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["shared", "only Furnace1", "only Furnace2"],
        [len(shared), len(only1), len(only2)],
        color=["#27ae60", "#2980b9", "#c0392b"],
    )
    ax.set_ylabel("feature count")
    ax.set_title("Feature overlap between clients")
    fig.savefig(FIG_DIR / "feature_overlap.png")
    plt.close(fig)
    return {"shared": shared, "only_furnace1": only1, "only_furnace2": only2}


def describe_csv(name: str, df: pd.DataFrame) -> None:
    df.describe().T.to_csv(FIG_DIR / f"{name}_describe.csv")


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}

    for name, path in DATASETS.items():
        df = load(name, path)
        frames[name] = df
        summaries[name] = summarise(name, df)
        describe_csv(name, df)
        plot_missing(name, df)
        plot_distributions(name, df)
        plot_correlation(name, df)
        plot_timeseries(
            name,
            df,
            ["active_power", "voltage_a", "current_a", "hearth_temp_1"],
        )
        engineered = feature_engineer(df)
        plot_engineered(name, engineered)
        plot_pca(name, df)

    overlap = plot_feature_overlap(frames)
    summaries["overlap"] = overlap

    (FIG_DIR / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
