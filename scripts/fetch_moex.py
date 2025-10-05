from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import List, Optional

import requests
from tqdm import tqdm

from utils import MOEX_BASE, complete_data, get_moex_history

MIN_START_DATE = datetime(1990, 1, 1)






def fetch_tqbr_secids(limit: Optional[int], offset: int) -> List[str]:
    """Fetch SECID list for the TQBR board in a single request."""
    url = f"{MOEX_BASE}/engines/stock/markets/shares/boards/TQBR/securities.json"
    session = requests.Session()

    offset = max(offset, 0)
    limit_value: Optional[int] = None if limit is None or limit <= 0 else int(limit)

    def moex_request(params: dict) -> dict:
        for attempt in range(5):
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:  # pragma: no cover - network issues
                wait = 1.5 * (attempt + 1)
                print(f"[fetch_tqbr_secids] request failed ({exc.__class__.__name__}), retry in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError("MOEX request failed after retries")

    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "securities.columns": "SECID",
        "securities.limit": 0,
    }
    payload = moex_request(params)
    table = payload.get("securities", {})
    rows = table.get("data", [])
    secids_all = [str(row[0]) for row in rows if row and row[0]]

    total_available = len(secids_all)
    print(f"Total TQBR securities: {total_available}")

    if offset >= total_available:
        return []

    if limit_value is not None:
        secids = secids_all[offset: offset + limit_value]
    else:
        secids = secids_all[offset:]

    progress = tqdm(total=len(secids), desc="Discovering TQBR", unit="tick", leave=False)
    collected: List[str] = []
    for secid in secids:
        collected.append(secid)
        progress.update(1)
        remaining = max(len(secids) - len(collected), 0)
        progress.set_postfix({"left": remaining})
    progress.close()
    return collected

def parse_args() -> argparse.Namespace:
    def parse_date(value: str) -> datetime:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value}") from exc

    parser = argparse.ArgumentParser(
        description="Download MOEX historical candles for TQBR securities (simple version).",
    )
    parser.add_argument("--start-date", type=parse_date, help="First trading day (YYYY-MM-DD).", default=None)
    parser.add_argument("--end-date", type=parse_date, help="Last trading day (YYYY-MM-DD).", default=None)
    parser.add_argument("--output-dir", default="./data/tqbr", help="Directory for CSV output.")
    parser.add_argument("--limit", type=int, default=10, help="How many tickers to download (<=0 for all).")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N tickers from the list.")
    parser.add_argument("--pause", type=float, default=0.1, help="Pause between tickers in seconds.")
    parser.add_argument("--sleep", type=float, default=0.35, help="Pause between MOEX pages when downloading history.")
    parser.add_argument("--force", action="store_true", help="Ignore cached CSV files.")
    parser.add_argument("--min-rows-cache", type=int, default=1, help="Minimal rows required to reuse cached CSV.")
    parser.add_argument("--max-age-cache", type=int, default=30, help="Max cache age in days before refresh.")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected tickers without downloading.")
    parser.add_argument("--no-clean", action="store_true", help="Skip post-processing with complete_data().")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    end_dt = args.end_date or datetime.now()
    start_dt = args.start_date or MIN_START_DATE
    if start_dt > end_dt:
        raise SystemExit("start-date must not be after end-date")

    limit = None if args.limit is None or args.limit <= 0 else int(args.limit)
    secids = fetch_tqbr_secids(limit=limit, offset=args.offset)
    if not secids:
        print("No securities returned by MOEX for the selected parameters.")
        return

    print(f"Securities selected: {len(secids)} (offset={args.offset}, limit={'all' if limit is None else limit}).")
    print(f"Date range: {start_dt:%Y-%m-%d} -> {end_dt:%Y-%m-%d}.")

    if args.dry_run:
        preview = ", ".join(secids[:min(10, len(secids))])
        print("Dry run enabled. Sample tickers: " + preview)
        if len(secids) > 10:
            print(f"... and {len(secids) - 10} more")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    successes = 0
    empty = 0
    failures = 0

    prog = tqdm(secids, desc="Tickers", unit="tick", leave=True)
    for idx, secid in enumerate(prog, start=1):
        out_path = os.path.join(args.output_dir, f"{secid}.csv")
        try:
            df = get_moex_history(
                ticker=secid,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                out_dir=args.output_dir,
                board="TQBR",
                sleep_sec=args.sleep,
                force=args.force,
                min_rows_cache=args.min_rows_cache,
                max_age_days_cache=args.max_age_cache,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            prog.write(f"{secid}: ERROR {exc}")
            remaining = max(len(secids) - idx, 0)
            prog.set_postfix({"left": remaining})
            continue

        if df.empty and not os.path.exists(out_path):
            empty += 1
            prog.write(f"{secid}: no rows returned")
        else:
            if not args.no_clean and os.path.exists(out_path):
                cleaned = complete_data(out_path)
                rows = 0 if cleaned.empty else len(cleaned)
            else:
                rows = len(df)
            if rows == 0:
                empty += 1
                prog.write(f"{secid}: file empty after processing")
            else:
                successes += 1
                prog.write(f"{secid}: saved {rows} rows -> {out_path}")

        if args.pause > 0:
            time.sleep(args.pause)

        remaining = max(len(secids) - idx, 0)
        prog.set_postfix({"left": remaining})

    prog.close()
    print(f"Done. Success: {successes}, empty: {empty}, errors: {failures}.")


if __name__ == "__main__":
    main()
