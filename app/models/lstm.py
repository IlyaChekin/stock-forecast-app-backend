from __future__ import annotations

import glob, os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

from .registry import latest_dir, save_json, ensure_dirs


@dataclass
class LSTMParams:
    n_lags: int = 30
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 15
    test_ratio: float = 0.25   # 3:1 → тест 25%
    corpus_dir: str = "./data"


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # (B, T, H)
        last = out[:, -1, :]       # (B, H)
        y = self.fc(last)          # (B, 1)
        return y


# ---------- utils ----------

def _prices_to_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    p = np.maximum(p, 1e-12)
    logp = np.log(p)
    return np.diff(logp)  # r_t = log P_t − log P_{t-1}

def _make_seq(ret: np.ndarray, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строим обучающие пары (X_t -> y_t), где
      X_t = [r_{t-n_lags}, ..., r_{t-1}],  y_t = r_t.
    Для ret длины R число пар = R - n_lags.
    """
    ret = np.asarray(ret, dtype=np.float32)
    R = len(ret)
    if R <= n_lags:
        return np.empty((0, n_lags, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Окна ретёрнов: shape (R - n_lags + 1, n_lags) — отрезаем последнюю,
    # чтобы пар ровно совпадало с y = ret[n_lags:] (длина R - n_lags).
    X = np.lib.stride_tricks.sliding_window_view(ret, n_lags)[:-1]
    X = X.reshape(-1, n_lags, 1).astype(np.float32)

    y = ret[n_lags:].astype(np.float32)  # длина R - n_lags

    # страховка (не должна срабатывать теперь)
    if len(X) != len(y):
        m = min(len(X), len(y))
        X, y = X[:m], y[:m]
        print(f"[warn] align _make_seq: X={len(X)} y={len(y)} (trimmed)")

    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}

def _returns_to_price_tf(prices: np.ndarray, price_idx: np.ndarray, r_pred: np.ndarray) -> np.ndarray:
    # P̂_t = P_{t-1} * exp(r̂_t), где price_idx[k] — индекс цены t
    out = np.zeros_like(r_pred, dtype=float)
    for k, idx in enumerate(price_idx):
        out[k] = prices[idx - 1] * np.exp(r_pred[k])
    return out


# ---------- TRAIN: по корпусу CSV, сплит 3:1 и сохранение артефактов ----------

def train_lstm_corpus_and_save(
    n_lags: int = 30,
    epochs: int = 15,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 64,
    corpus_dir: str = "./dataset/train",
    test_ratio: float = 0.25,
) -> Dict:
    """
    Читает ВСЕ CSV из corpus_dir (колонка Close), бьёт окна по returns, валидирует на 25%,
    обучает и сохраняет артефакты в models/lstm/latest.

    Прогресс-бар: tqdm по эпохам. Каждые 5 эпох — печать train/val MSE (в пространстве стандартизованных returns).
    """
    ensure_dirs()
    device = "cpu"

    # собрать серии
    series = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.csv"))):
        try:
            df = pd.read_csv(path)
            col = next((c for c in df.columns if c.lower() == "close"), None)
            dcol = next((c for c in df.columns if c.lower() in ("date","time","timestamp","tradedate")), None)
            if col is None:
                continue
            price = pd.to_numeric(df[col], errors="coerce").ffill().dropna().values
            if len(price) < (n_lags + 10):
                continue
            ret = _prices_to_returns(price)
            X_all, y_all = _make_seq(ret, n_lags)
            if len(X_all) < 30:
                continue
            split = int(len(X_all) * (1 - test_ratio))
            split = max(1, min(split, len(X_all) - 1))
            price_idx_all = np.arange(n_lags, n_lags + len(y_all))
            series.append(dict(
                path=path,
                price=price,
                dates=(pd.to_datetime(df[dcol]).astype(str).values if dcol else None),
                X_train=X_all[:split], y_train=y_all[:split], idx_train=price_idx_all[:split],
                X_test=X_all[split:],  y_test=y_all[split:],  idx_test=price_idx_all[split:],
            ))
        except Exception:
            continue

    if not series:
        raise ValueError(f"В {corpus_dir} нет пригодных CSV.")

    # скейлер по всем train returns
    scaler = StandardScaler()
    y_train_concat = np.concatenate([s["y_train"] for s in series]).reshape(-1,1)
    scaler.fit(y_train_concat)

    def scale_X(X):
        flat = X.reshape(-1,1)
        return scaler.transform(flat).reshape(X.shape)

    def scale_y(y):
        return scaler.transform(y.reshape(-1,1)).ravel()

    # train/val матрицы (scaled)
    X_train = np.concatenate([scale_X(s["X_train"]) for s in series], axis=0)
    y_train = np.concatenate([scale_y(s["y_train"]) for s in series], axis=0)
    X_val   = np.concatenate([scale_X(s["X_test"])  for s in series], axis=0)
    y_val   = np.concatenate([scale_y(s["y_test"])  for s in series], axis=0)

    # dataloaders
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    Xv_t = torch.tensor(X_val, dtype=torch.float32)
    yv_t = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32)

    # модель
    model = LSTMRegressor(1, hidden_size, num_layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- цикл обучения с tqdm ---
    pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        batches = 0

        for xb, yb in dl:
            opt.zero_grad()
            yhat = model(xb.to(device))
            loss = loss_fn(yhat, yb.to(device))
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            batches += 1

        avg_train_batch_loss = running_loss / max(1, batches)
        pbar.set_postfix({"batch_loss": f"{avg_train_batch_loss:.6f}"})

        # Каждые 5 эпох — полные train/val лоссы (на всём датасете)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                yhat_tr = model(X_t.to(device))
                full_train_loss = float(loss_fn(yhat_tr, y_t.to(device)).item())
                yhat_val = model(Xv_t.to(device))
                full_val_loss = float(loss_fn(yhat_val, yv_t.to(device)).item())
            print(f"[epoch {epoch:03d}] train_mse={full_train_loss:.6f} | val_mse={full_val_loss:.6f}")

    # === Валидация в ценах (TF/REC), с выравниванием длин (как раньше) ===
    model.eval()
    y_test_prices_true, y_test_prices_tf, y_test_prices_rec = [], [], []
    for s in series:
        # TF
        Xs = torch.tensor(scale_X(s["X_test"]), dtype=torch.float32)
        with torch.no_grad():
            yhat_s = model(Xs).cpu().numpy().ravel()
        yhat = scaler.inverse_transform(yhat_s.reshape(-1, 1)).ravel()

        price_tf = _returns_to_price_tf(s["price"], s["idx_test"], yhat)
        price_true = s["price"][s["idx_test"]]

        # REC
        last_returns = _prices_to_returns(s["price"])[: s["idx_train"][-1]]
        init = last_returns[-n_lags:]
        win = deque(scaler.transform(init.reshape(-1, 1)).ravel().tolist(), maxlen=n_lags)
        prev_p = s["price"][s["idx_train"][-1]]
        rec_prices = []
        for _ in range(len(s["idx_test"])):
            x = torch.tensor(np.array(win, dtype=np.float32).reshape(1, n_lags, 1))
            with torch.no_grad():
                rhat_s = model(x).cpu().numpy().ravel()[0]
            rhat = scaler.inverse_transform([[rhat_s]]).ravel()[0]
            next_p = prev_p * np.exp(rhat)
            rec_prices.append(next_p)
            prev_p = next_p
            win.append(rhat_s)
        rec_prices = np.asarray(rec_prices, dtype=float)

        # выравнивание длин
        m = min(len(price_true), len(price_tf), len(rec_prices))
        if (len(price_true) != len(price_tf)) or (len(price_true) != len(rec_prices)):
            print(f"[warn] length mismatch: {os.path.basename(s['path'])} "
                  f"true={len(price_true)} tf={len(price_tf)} rec={len(rec_prices)} -> cut to {m}")

        y_test_prices_true.append(price_true[:m])
        y_test_prices_tf.append(price_tf[:m])
        y_test_prices_rec.append(rec_prices[:m])

    y_true = np.concatenate(y_test_prices_true) if len(y_test_prices_true) else np.array([])
    y_tf   = np.concatenate(y_test_prices_tf)   if len(y_test_prices_tf)   else np.array([])
    y_rec  = np.concatenate(y_test_prices_rec)  if len(y_test_prices_rec)  else np.array([])

    m_tf  = _metrics(y_true, y_tf)
    m_rec = _metrics(y_true, y_rec)

    # сохранить артефакты
    out = latest_dir()
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    joblib.dump(scaler, os.path.join(out, "scaler.pkl"))
    save_json(os.path.join(out, "config.json"), dict(
        n_lags=n_lags, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        lr=lr, batch_size=batch_size, epochs=epochs, test_ratio=test_ratio,
        corpus_dir=os.path.abspath(corpus_dir)
    ))

    return {
        "artifact_dir": out,
        "series_used": len(series),
        "metrics": {
            "test_teacher_forced": m_tf,
            "test_recursive": m_rec,
        }
    }



# ---------- INFERENCE: применяем сохранённую модель к одному CSV ----------

def predict_with_pretrained_on_series(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
) -> Dict:
    out = latest_dir()
    model_path = os.path.join(out, "model.pt")
    scaler_path = os.path.join(out, "scaler.pkl")
    cfg_path = os.path.join(out, "config.json")
    if not (os.path.isfile(model_path) and os.path.isfile(scaler_path) and os.path.isfile(cfg_path)):
        raise RuntimeError("Нет предобученной LSTM. Сначала обучи модель (scripts/train_lstm.py или POST /api/train/lstm).")

    import json
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    n_lags = int(cfg.get("n_lags", 30))

    # данные
    if date_col and date_col in df.columns:
        df = df.sort_values(by=date_col).reset_index(drop=True)
    price = pd.to_numeric(df[target_col], errors="coerce").ffill().dropna().values
    if len(price) < (n_lags + 10):
        raise ValueError("Слишком короткий ряд для LSTM.")

    ret = _prices_to_returns(price)
    X_all, y_all = _make_seq(ret, n_lags)
    if len(X_all) < 30:
        raise ValueError("Недостаточно примеров после формирования окон.")

    split = int(len(X_all) * (1 - 0.25))  # фиксированный 3:1
    split = max(1, min(split, len(X_all) - 1))
    price_idx_all = np.arange(n_lags, n_lags + len(y_all))
    idx_train, idx_test = price_idx_all[:split], price_idx_all[split:]

    scaler: StandardScaler = joblib.load(scaler_path)
    model = LSTMRegressor(1, int(cfg["hidden_size"]), int(cfg["num_layers"]), float(cfg["dropout"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    def scale_X(X):
        flat = X.reshape(-1,1)
        return scaler.transform(flat).reshape(X.shape)

    # train preds
    with torch.no_grad():
        y_train_hat_s = model(torch.tensor(scale_X(X_all[:split]), dtype=torch.float32)).cpu().numpy().ravel()
    y_train_hat = scaler.inverse_transform(y_train_hat_s.reshape(-1,1)).ravel()
    y_train_price_true = price[idx_train]
    y_train_price_pred = _returns_to_price_tf(price, idx_train, y_train_hat)

    # test TF
    with torch.no_grad():
        y_test_hat_s = model(torch.tensor(scale_X(X_all[split:]), dtype=torch.float32)).cpu().numpy().ravel()
    y_test_hat = scaler.inverse_transform(y_test_hat_s.reshape(-1,1)).ravel()
    y_test_price_true = price[idx_test]
    y_test_price_pred_tf = _returns_to_price_tf(price, idx_test, y_test_hat)

    # test REC
    last_returns = ret[idx_train[-1]-n_lags: idx_train[-1]]
    win = deque(scaler.transform(last_returns.reshape(-1,1)).ravel().tolist(), maxlen=n_lags)
    prev_p = price[idx_train[-1]]
    rec_prices = []
    for _ in range(len(idx_test)):
        x = torch.tensor(np.array(win, dtype=np.float32).reshape(1, n_lags, 1))
        with torch.no_grad():
            rhat_s = model(x).cpu().numpy().ravel()[0]
        rhat = scaler.inverse_transform([[rhat_s]]).ravel()[0]
        next_p = prev_p * np.exp(rhat)
        rec_prices.append(next_p)
        prev_p = next_p
        win.append(rhat_s)
    y_pred_rec = np.asarray(rec_prices, dtype=float)

    # наивные
    naive_tf = np.empty_like(y_test_price_true)
    naive_tf[0] = y_train_price_true[-1]
    if len(y_test_price_true) > 1:
        naive_tf[1:] = y_test_price_true[:-1]
    naive_rec = np.full_like(y_test_price_true, y_train_price_true[-1], dtype=float)

    m_train = _metrics(y_train_price_true, y_train_price_pred)
    m_tf    = _metrics(y_test_price_true, y_test_price_pred_tf)
    m_rec   = _metrics(y_test_price_true, y_pred_rec)
    m_ntf   = _metrics(y_test_price_true, naive_tf)
    m_nrc   = _metrics(y_test_price_true, naive_rec)

    # таймстемпы
    if date_col and date_col in df.columns:
        ts = pd.to_datetime(df[date_col]).astype(str).values
        ts_train = ts[idx_train].tolist()
        ts_test  = ts[idx_test].tolist()
    else:
        ts_train = idx_train.tolist()
        ts_test  = idx_test.tolist()

    return {
        "target_col": target_col,
        "n_lags": n_lags,
        "test_size": 0.25,
        "model_key": "lstm",
        "model_name": "LSTM (pretrained, returns)",
        "metrics": {
            "train": m_train,
            "teacher_forced": m_tf,
            "recursive": m_rec,
            "naive_teacher_forced": m_ntf,
            "naive_recursive": m_nrc,
            "improve_tf_vs_naive_tf_pct": float((m_ntf["rmse"] - m_tf["rmse"])/(m_ntf["rmse"]+1e-12)*100),
            "improve_rec_vs_naive_rec_pct": float((m_nrc["rmse"] - m_rec["rmse"])/(m_nrc["rmse"]+1e-12)*100),
        },
        "coefficients": None,
        "timestamps": {"train": ts_train, "test": ts_test},
        "series": {
            "y_train": y_train_price_true.tolist(),
            "y_train_pred": y_train_price_pred.tolist(),
            "y_test": y_test_price_true.tolist(),
            "y_pred_teacher_forced": y_test_price_pred_tf.tolist(),
            "y_pred_recursive": y_pred_rec.tolist(),
            "y_naive_teacher_forced": naive_tf.tolist(),
            "y_naive_recursive": naive_rec.tolist(),
        },
    }
