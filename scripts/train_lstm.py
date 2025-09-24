# scripts/train_lstm.py
from __future__ import annotations
import argparse
from app.models.lstm import train_lstm_corpus_and_save

def main():
    ap = argparse.ArgumentParser(description="Обучение LSTM на ./dataset/train (3:1 внутри каждой серии при формировании окон).")
    ap.add_argument("--corpus", default="./dataset/train", help="папка с train CSV (после split_dataset)")
    ap.add_argument("--n-lags", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    res = train_lstm_corpus_and_save(
        n_lags=args.n_lags,
        epochs=args.epochs,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch,
        corpus_dir=args.corpus,
        test_ratio=0.25,  # фиксируем 3:1
    )

    print("\n=== Артефакты ===")
    print("Папка:", res["artifact_dir"])
    print("Серий использовано:", res["series_used"])
    print("TEST TF:", res["metrics"]["test_teacher_forced"])
    print("TEST REC:", res["metrics"]["test_recursive"])

if __name__ == "__main__":
    main()
