"""Download and prepare the NSL-KDD dataset."""

import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}
RAW_FILES = {
    "train": DATA_DIR / "KDDTrain+.txt",
    "test": DATA_DIR / "KDDTest+.txt",
}
CSV_FILES = {
    "train": DATA_DIR / "train.csv",
    "test": DATA_DIR / "test.csv",
}

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already cached: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print("done")


def load_and_save(split: str) -> pd.DataFrame:
    raw = RAW_FILES[split]
    csv = CSV_FILES[split]
    df = pd.read_csv(raw, header=None, names=COLUMNS)
    df.to_csv(csv, index=False)
    return df


def print_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print("  Class distribution:")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        print(f"    {label:<30} {count:>7,}  ({100*count/len(df):.1f}%)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading NSL-KDD dataset files...")
    for split, url in URLS.items():
        download_file(url, RAW_FILES[split])

    print("\nProcessing files...")
    train_df = load_and_save("train")
    test_df = load_and_save("test")

    print_summary("Train set", train_df)
    print_summary("Test set", test_df)
    print(f"\nSaved: {CSV_FILES['train']}")
    print(f"Saved: {CSV_FILES['test']}")


if __name__ == "__main__":
    main()
