import io
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ml
from .data import CATS, generate

app = FastAPI(title="Adaptive Hybrid Ensemble Fraud Detection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE: dict = {"raw": None, "result": None}
LOCK = threading.Lock()


def _train(raw: pd.DataFrame):
    with LOCK:
        STATE["raw"], STATE["result"] = raw, ml.run_pipeline(raw)


def _need():
    if STATE["result"] is None:
        raise HTTPException(400, "Model not trained yet. Call POST /api/train first.")
    return STATE["result"]


@app.get("/api/health")
def health():
    return {"status": "ok", "trained": STATE["result"] is not None}


@app.post("/api/train")
def train_synthetic():
    _train(generate())
    return {"ok": True, "alpha": STATE["result"]["alpha"], **STATE["result"]["summary"]}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """CSV columns: card_id, timestamp, amount, merchant_category, distance_km, is_fraud"""
    df = pd.read_csv(io.BytesIO(file.file.read()))
    missing = {"card_id", "timestamp", "amount", "merchant_category", "is_fraud"} - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {sorted(missing)}")
    _train(df)
    return {"ok": True, "alpha": STATE["result"]["alpha"], **STATE["result"]["summary"]}


@app.get("/api/results")
def results():
    r = _need()
    return {"alpha": r["alpha"], "beta": round(1 - r["alpha"], 2), "summary": r["summary"], "metrics": r["metrics"]}


@app.get("/api/flagged")
def flagged(limit: int = 30):
    t = _need()["test"]
    top = t[t.flagged == 1].sort_values("final", ascending=False).head(limit)
    cols = ["card_id", "timestamp", "amount", "merchant_category", "distance_km", "iforest",
            "autoencoder", "lof", "ensemble", "personal", "final", "is_fraud"]
    out = top[cols].copy()
    out["id"] = top.index
    out["timestamp"] = out["timestamp"].astype(str)
    return out.round(3).to_dict("records")


def _explain_row(card_id, amount, hour, dist, t1, t24, cat):
    eng = _need()["engine"]
    return eng.explain(card_id, [amount, hour, dist, t1, t24, ml.CAT_IDX[cat]])


@app.get("/api/explain/{txn_id}")
def explain(txn_id: int):
    t = _need()["test"]
    if txn_id not in t.index:
        raise HTTPException(404, "Unknown transaction id")
    r = t.loc[txn_id]
    return _explain_row(r.card_id, r.amount, r.hour, r.distance_km, r.txn_1h, r.txn_24h, r.merchant_category)


class Txn(BaseModel):
    card_id: str
    amount: float
    hour: int
    merchant_category: str
    distance_km: float = 0
    txn_1h: int = 1
    txn_24h: int = 1


@app.post("/api/score")
def score(t: Txn):
    r = _need()
    if t.merchant_category not in CATS:
        raise HTTPException(400, f"merchant_category must be one of {CATS}")
    eng = r["engine"]
    s = eng.score_arrays(eng.lookup([t.card_id]), np.log1p([t.amount]), np.array([t.hour]),
                         np.array([t.distance_km]), np.array([t.txn_1h]), np.array([t.txn_24h]),
                         np.array([ml.CAT_IDX[t.merchant_category]]))
    return {"final_score": round(float(s["final"][0]), 4), "threshold": eng.threshold,
            "flagged": bool(s["final"][0] >= eng.threshold),
            "components": {k: round(float(v[0]), 4) for k, v in s.items()},
            "explanation": _explain_row(t.card_id, t.amount, t.hour, t.distance_km, t.txn_1h, t.txn_24h,
                                        t.merchant_category)}


@app.get("/api/drift")
def drift():
    _need()
    return ml.drift_experiment(STATE["raw"])


@app.get("/api/categories")
def categories():
    return CATS


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[2] / "frontend", html=True), name="ui")
