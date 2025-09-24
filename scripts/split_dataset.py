# scripts/split_dataset.py
from __future__ import annotations
import argparse, os, glob
import pandas as pd

DATE_CAND = ("date", "tradedate", "time", "timestamp")
CLOSE_CAND = ("close", "price", "close_price", "adj_close")

def detect_col(df: pd.DataFrame, cands: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in cands:
        if name in lower:
            return lower[name]
    return None

def load_and_standardize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dcol = detect_col(df, DATE_CAND)
    ccol = detect_col(df, CLOSE_CAND)
    if ccol is None:
        raise ValueError(f"{os.path.basename(path)}: не найден столбец цены (Close/Price ...)")
    out = pd.DataFrame()
    if dcol:
        out["Date"] = pd.to_datetime(df[dcol], errors="coerce")
    out["Close"] = pd.to_numeric(df[ccol], errors="coerce")
    out = out.dropna(subset=["Close"])
    if "Date" in out.columns:
        out = out.dropna(subset=["Date"]).sort_values("Date")
    out = out.reset_index(drop=True)
    return out[["Date","Close"]] if "Date" in out.columns else out[["Close"]]

def split_timewise(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 10:
        raise ValueError("слишком мало строк (<10)")
    split = int(n * (1 - test_ratio))
    split = max(1, min(split, n - 1))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def main():
    ap = argparse.ArgumentParser(description="Временной сплит всех CSV из ./data на ./dataset/train и ./dataset/test (3:1).")
    ap.add_argument("--src", default="./data", help="папка с исходными CSV")
    ap.add_argument("--out", default="./dataset", help="куда положить train/test")
    ap.add_argument("--test-ratio", type=float, default=0.25, help="доля теста (по времени), по умолчанию 0.25")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test"),  exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "*.csv")))
    if not files:
        print(f"В {args.src} не найдено CSV.")
        return

    ok, skipped = 0, 0
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_and_standardize(path)
            tr, te = split_timewise(df, test_ratio=args.test_ratio)
            tr.to_csv(os.path.join(args.out, "train", f"{name}.csv"), index=False)
            te.to_csv(os.path.join(args.out, "test",  f"{name}.csv"), index=False)
            print(f"{name}: train={len(tr)}  test={len(te)}")
            ok += 1
        except Exception as e:
            print(f"{name}: SKIP ({e})")
            skipped += 1

    print(f"\nГотово: ок={ok}, пропущено={skipped}.")
    print(f"Train → {os.path.abspath(os.path.join(args.out, 'train'))}")
    print(f"Test  → {os.path.abspath(os.path.join(args.out, 'test'))}")

if __name__ == "__main__":
    main()
