import io
import pandas as pd
from fastapi import UploadFile

async def read_csv_to_df(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    buf = io.BytesIO(content)
    try:
        df = pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, sep=";")
    # Попытка парсинга дат для колонок, похожих на даты
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

def detect_date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        name = c.lower()
        if "date" in name or "time" in name or "timestamp" in name:
            return c
    # Иногда дата в индексе — тут ничего не делаем, вернём None
    return None

def detect_target_col(df: pd.DataFrame) -> str:
    # Частые имена для цены
    candidates = ["close", "adj_close", "price", "value", "close_price"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    # Иначе — первая числовая
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    # Фоллбэк — первая колонка
    return df.columns[0]
