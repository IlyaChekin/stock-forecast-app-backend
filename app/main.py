# app/main.py
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .inference import run_any_model
from .utils import read_csv_to_df, detect_date_col, detect_target_col

# СОЗДАЁМ ПРИЛОЖЕНИЕ СРАЗУ, ДО ЛЮБЫХ ДЕКОРАТОРОВ
app = FastAPI(title="Stock Forecast Demo", version="0.4.0")

# Статика
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.get("/favicon.ico")
def favicon():
    # можно положить файл в /app/static/favicon.ico и убрать эту заглушку
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# --------- Предсказание (классика тренируется на лету, LSTM - предобученная) ----------
@app.post("/api/predict/regression")
async def predict_regression(
    file: UploadFile = File(...),
    model: str = Form("linreg"),
    target: Optional[str] = Form(None),
    date_col: Optional[str] = Form(None),
    n_lags: int = Form(10),
    test_size: float = Form(0.2),
):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Принимаются только CSV файлы.")

        df = await read_csv_to_df(file)
        if df.empty:
            raise HTTPException(status_code=400, detail="Пустой или некорректный CSV.")

        date_col = date_col or detect_date_col(df)
        target = target or detect_target_col(df)
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Не найден target-столбец '{target}'.")

        # LSTM использует предобученный чекпоинт; n_lags/test_size применимы только к классике
        result = run_any_model(
            df=df,
            target_col=target,
            date_col=date_col,
            model_key=model,
            n_lags=n_lags,
            test_size=test_size,
        )
        return JSONResponse(result)

    except RuntimeError as e:
        # например, нет артефактов LSTM в models/lstm/latest
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
