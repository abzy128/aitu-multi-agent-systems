from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fl_saf.config import DataConfig

TARGET = "active_power"
SHARED_MULTIVARIATE_COLUMNS = [
    "active_power",
    "reactive_power",
    "voltage_a",
    "voltage_b",
    "voltage_c",
    "current_a",
    "current_b",
    "current_c",
    "electrode_position_a",
    "electrode_position_b",
    "electrode_position_c",
    "electrode_slip_a",
    "electrode_slip_b",
    "electrode_slip_c",
    "hearth_temp_1",
    "water_temp_to_furnace_1",
    "water_temp_to_furnace_2",
    "water_after_cooling_1",
    "water_after_cooling_2",
]


@dataclass(slots=True)
class PreprocessingReport:
    client_id: str
    track: str
    feature_names: list[str]
    dropped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scaler_mean: list[float] = field(default_factory=list)
    scaler_scale: list[float] = field(default_factory=list)
    target_mean: float = 0.0
    target_scale: float = 1.0


def load_client_csv(path: str, client_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DateTime"]).sort_values("DateTime")
    df = df.reset_index(drop=True)
    diffs = df["DateTime"].diff().dropna().dt.total_seconds()
    if not diffs.empty and not np.allclose(diffs.to_numpy(), 60.0):
        counts = diffs.value_counts().head(5).to_dict()
        minute_ratio = float((diffs == 60.0).mean())
        if minute_ratio < 0.99:
            raise ValueError(f"{client_id} is not on a uniform 60 s cadence: {counts}")
        df.attrs["cadence_warnings"] = [f"rare non-60s timestamp gaps detected: {counts}"]
    return df


def select_features(
    df: pd.DataFrame, cfg: DataConfig, client_id: str
) -> tuple[pd.DataFrame, PreprocessingReport]:
    if cfg.track == "univariate":
        feature_names = [TARGET]
        out = df[["DateTime", TARGET]].copy()
        report = PreprocessingReport(client_id, cfg.track, feature_names)
        report.warnings.extend(df.attrs.get("cadence_warnings", []))
        return out, report

    missing = [c for c in SHARED_MULTIVARIATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{client_id} missing shared columns: {missing}")

    out = df[["DateTime", *SHARED_MULTIVARIATE_COLUMNS]].copy()
    numeric_cols = [c for c in SHARED_MULTIVARIATE_COLUMNS if c != TARGET]
    dropped = [c for c in numeric_cols if float(out[c].std()) < cfg.constant_eps]
    keep = [TARGET, *[c for c in numeric_cols if c not in dropped]]
    report = PreprocessingReport(client_id, cfg.track, keep, dropped_columns=dropped)
    report.warnings.extend(df.attrs.get("cadence_warnings", []))
    report.warnings.extend(calibration_warnings(out, client_id))
    return out[["DateTime", *keep]], report


def calibration_warnings(df: pd.DataFrame, client_id: str) -> list[str]:
    warnings: list[str] = []
    if client_id != "client_2":
        return warnings
    va, vb, vc = df["voltage_a"].mean(), df["voltage_b"].mean(), df["voltage_c"].mean()
    ref_v = (abs(va) + abs(vb)) / 2
    if ref_v > 0 and abs(vc) / ref_v > 10:
        warnings.append("voltage_c mean differs from voltage_a/b by >10x")
    ca, cb = abs(df["current_a"].mean()), abs(df["current_b"].mean())
    if min(ca, cb) > 0 and max(ca, cb) / min(ca, cb) > 10:
        warnings.append("current_a/current_b means differ by >10x")
    if "cos_phi" in df and ((df["cos_phi"] < -1) | (df["cos_phi"] > 1)).any():
        warnings.append("cos_phi has values outside [-1, 1]")
    return warnings


def fit_transform_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    report: PreprocessingReport,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PreprocessingReport]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[feature_names]).astype(np.float32)
    val_x = scaler.transform(val[feature_names]).astype(np.float32)
    test_x = scaler.transform(test[feature_names]).astype(np.float32)
    report.scaler_mean = [float(x) for x in scaler.mean_]
    report.scaler_scale = [float(x) for x in scaler.scale_]
    target_idx = feature_names.index(TARGET)
    report.target_mean = report.scaler_mean[target_idx]
    report.target_scale = report.scaler_scale[target_idx]
    return train_x, val_x, test_x, scaler, report


def make_windows(
    values: np.ndarray,
    target_index: int,
    seq_len: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    last_start = len(values) - seq_len - horizon + 1
    for start in range(0, max(0, last_start), stride):
        end = start + seq_len
        xs.append(values[start:end])
        ys.append(values[end : end + horizon, target_index])
    if not xs:
        raise ValueError("not enough rows to create sequence windows")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
