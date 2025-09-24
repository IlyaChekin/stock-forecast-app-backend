# backend/scripts/fetch_moex.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

from utils import get_moex_history, complete_data, has_fresh_file


def main():
    # Blue chips
    blue_chips = [
        "SBER", "GAZP", "LKOH", "ROSN", "MGNT",
        "NLMK", "GMKN", "SNGS", "PLZL", "TATN"
    ]

    # Growth (скорректируешь при необходимости; у Тинькофф обычно TCSG)
    growing_companies = [
        "YNDX", "TCSG", "OZON", "FIVE", "IRKT",
        "PIKK", "AQUA", "CBOM", "POSI", "RUSI"
    ]

    all_tickers = blue_chips + growing_companies

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    os.makedirs("./data", exist_ok=True)

    # 1) Скачиваем (пропуская свежие файлы)
    for ticker in all_tickers:
        try:
            out_path = os.path.join("./data", f"{ticker}.csv")
            if has_fresh_file(out_path, min_rows=200, max_age_days=7):
                print(f"Skip {ticker}: fresh file exists -> {out_path}")
                continue

            print(f"Downloading {ticker} [{start_date}..{end_date}]")
            df = get_moex_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                out_dir="./data",
                force=False,           # поставить True, чтобы перекачать при любой «свежести»
            )
            if df.empty:
                print(f"  -> no data for {ticker}")
            else:
                print(f"  -> saved {len(df)} rows to ./data/{ticker}.csv")

        except Exception as e:
            # Не валим весь процесс из-за одного тикера
            print(f"  -> ERROR on {ticker}: {e}. Skipping...")

        time.sleep(0.4)  # бережно относимся к API

    # 2) Докомплектация уже скачанных файлов (унификация колонок)
    directory = os.path.join(".", "data")
    for fname in os.listdir(directory):
        if fname.lower().endswith(".csv"):
            path = os.path.join(directory, fname)
            try:
                print(f"Completing {fname}")
                complete_data(path, None)
            except Exception as e:
                print(f"  -> complete_data error on {fname}: {e}")


if __name__ == "__main__":
    main()
