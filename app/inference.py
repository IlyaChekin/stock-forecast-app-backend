# app/inference.py
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _build_lag_features(df: pd.DataFrame, target_col: str, n_lags: int) -> pd.DataFrame:
    Xy = df[[target_col]].copy()
    for l in range(1, n_lags + 1):
        Xy[f"{target_col}_lag{l}"] = Xy[target_col].shift(l)
    return Xy.dropna()

def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}

def _get_model(model_key: str):
    key = (model_key or "linreg").lower()
    if key == "ridge":
        return Ridge(alpha=1.0, random_state=42), True, "Ridge"
    if key == "lasso":
        return Lasso(alpha=0.001, max_iter=10000, random_state=42), True, "Lasso"
    if key == "elasticnet":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42), True, "ElasticNet"
    if key == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42), False, "RandomForest"
    return LinearRegression(), True, "LinearRegression"

def _run_classic(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
    model_key: str,
    n_lags: int,
    test_size: float,
):
    if date_col and date_col in df.columns:
        df = df.sort_values(by=date_col).reset_index(drop=True)

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill()

    Xy = _build_lag_features(df, target_col, n_lags)
    feature_cols = [c for c in Xy.columns if c != target_col]
    X = Xy[feature_cols].values
    y = Xy[target_col].values
    if len(X) < 30:
        raise ValueError("Слишком мало данных после формирования лагов (нужно ≥30 наблюдений).")

    split_idx = int(len(X) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(X) - 1))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model, needs_scaling, model_name = _get_model(model_key)
    scaler = StandardScaler() if needs_scaling else None
    if scaler is not None:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test

    model.fit(X_train_s, y_train)

    # train (in-sample)
    y_train_pred = model.predict(X_train_s)

    # test: TF
    y_pred_tf = model.predict(X_test_s)

    # test: REC (окно из последних фактов тренировки)
    window = deque([y_train[-i] for i in range(1, n_lags + 1)], maxlen=n_lags)
    y_pred_rec = []
    for _ in range(len(y_test)):
        x = np.array(list(window))[None, :]
        if scaler is not None:
            x = scaler.transform(x)
        yhat = float(model.predict(x)[0])
        y_pred_rec.append(yhat)
        window.appendleft(yhat)
    y_pred_rec = np.asarray(y_pred_rec, dtype=float)

    # бенчмарки
    naive_tf = np.empty_like(y_test)
    naive_tf[0] = y_train[-1]
    if len(y_test) > 1:
        naive_tf[1:] = y_test[:-1]
    naive_rec = np.full_like(y_test, y_train[-1], dtype=float)

    m_train = _metrics(y_train, y_train_pred)
    m_tf    = _metrics(y_test, y_pred_tf)
    m_rec   = _metrics(y_test, y_pred_rec)
    m_ntf   = _metrics(y_test, naive_tf)
    m_nrc   = _metrics(y_test, naive_rec)

    improve_tf = float((m_ntf["rmse"] - m_tf["rmse"]) / (m_ntf["rmse"] + 1e-12) * 100)
    improve_rec = float((m_nrc["rmse"] - m_rec["rmse"]) / (m_nrc["rmse"] + 1e-12) * 100)

    if date_col and date_col in df.columns:
        idx = Xy.index
        ts_all = df[date_col].iloc[idx].astype(str).tolist()
        ts_train = ts_all[:split_idx]
        ts_test  = ts_all[split_idx:]
    else:
        ts_train = list(range(len(y_train)))
        ts_test  = list(range(len(y_test)))

    coefs = None
    if hasattr(model, "coef_"):
        coefs = {name: float(w) for name, w in zip(feature_cols, model.coef_)}

    return {
        "target_col": target_col,
        "n_lags": n_lags,
        "test_size": test_size,
        "model_key": model_key,
        "model_name": model_name,
        "metrics": {
            "train": m_train,
            "teacher_forced": m_tf,
            "recursive": m_rec,
            "naive_teacher_forced": m_ntf,
            "naive_recursive": m_nrc,
            "improve_tf_vs_naive_tf_pct": improve_tf,
            "improve_rec_vs_naive_rec_pct": improve_rec,
        },
        "coefficients": coefs,
        "timestamps": {"train": ts_train, "test": ts_test},
        "series": {
            "y_train": y_train.tolist(),
            "y_train_pred": y_train_pred.tolist(),
            "y_test": y_test.tolist(),
            "y_pred_teacher_forced": y_pred_tf.tolist(),
            "y_pred_recursive": y_pred_rec.tolist(),
            "y_naive_teacher_forced": naive_tf.tolist(),
            "y_naive_recursive": naive_rec.tolist(),
        },
    }


def run_any_model(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
    model_key: str,
    n_lags: int,
    test_size: float,
):
    key = (model_key or "linreg").lower()
    if key == "lstm":
        # предобученная LSTM: грузим чекпоинт и предсказываем
        from .models.lstm import predict_with_pretrained_on_series
        return predict_with_pretrained_on_series(
            df=df,
            target_col=target_col,
            date_col=date_col,
        )
    # классические модели — как раньше
    return _run_classic(df, target_col, date_col, key, n_lags, test_size)
