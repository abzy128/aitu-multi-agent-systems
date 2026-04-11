"""Preprocessing pipeline: encoding, scaling, label creation, train/val split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    CLASS_NAMES,
    LABEL_TO_CATEGORY,
    RANDOM_SEED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_SPLIT,
)

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_raw(split: str = "train") -> pd.DataFrame:
    path = TRAIN_CSV if split == "train" else TEST_CSV
    return pd.read_csv(path)


def _encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    encoders: dict[str, LabelEncoder] = {}
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]], axis=0))
        train_out[col] = le.transform(train_df[col])
        test_out[col] = le.transform(test_df[col].map(
            lambda x, le=le: x if x in le.classes_ else le.classes_[0]
        ))
        encoders[col] = le
    return train_out, test_out, encoders


def _map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["label"].str.lower().map(LABEL_TO_CATEGORY).fillna("DoS")
    df["binary_label"] = (df["category"] != "normal").astype(int)
    df["class_label"] = df["category"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    return df


def build_pipeline(
    val_split: float = VAL_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        X_train, X_val, X_test          : float32 numpy arrays  (features)
        y_bin_train, y_bin_val, y_bin_test   : int64 arrays  (binary: 0=normal, 1=attack)
        y_cls_train, y_cls_val, y_cls_test   : int64 arrays  (5-class)
        feature_names                   : list[str]
        scaler                          : fitted StandardScaler
        encoders                        : dict[col → LabelEncoder]
        normal_mask_train               : bool array (rows where y_bin == 0, for AE training)
    """
    train_raw = load_raw("train")
    test_raw = load_raw("test")

    train_enc, test_enc, encoders = _encode_categoricals(train_raw, test_raw)
    train_enc = _map_labels(train_enc)
    test_enc = _map_labels(test_enc)

    feature_cols = [c for c in train_enc.columns
                    if c not in ("label", "difficulty", "category", "binary_label", "class_label")]

    X_all = train_enc[feature_cols].values.astype(np.float32)
    y_bin_all = train_enc["binary_label"].values.astype(np.int64)
    y_cls_all = train_enc["class_label"].values.astype(np.int64)

    X_train, X_val, y_bin_train, y_bin_val, y_cls_train, y_cls_val = train_test_split(
        X_all, y_bin_all, y_cls_all,
        test_size=val_split,
        random_state=random_seed,
        stratify=y_cls_all,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    X_test = test_enc[feature_cols].values.astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_bin_test = test_enc["binary_label"].values.astype(np.int64)
    y_cls_test = test_enc["class_label"].values.astype(np.int64)

    normal_mask_train = y_bin_train == 0

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_bin_train": y_bin_train,
        "y_bin_val": y_bin_val,
        "y_bin_test": y_bin_test,
        "y_cls_train": y_cls_train,
        "y_cls_val": y_cls_val,
        "y_cls_test": y_cls_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "encoders": encoders,
        "normal_mask_train": normal_mask_train,
    }
