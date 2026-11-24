"""Stage 1 ensemble + Stage 2 per-cardholder profiling + Stage 3 SHAP + Stage 4 drift."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_recall_fscore_support)
from .data import CATS

CAT_IDX = {c: i for i, c in enumerate(CATS)}
EXPL = ["amount", "hour", "distance_km", "txn_1h", "txn_24h", "merchant_category"]


# ---------------------------------------------------------------- preprocessing
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning + behavioural feature engineering (frequency 1h/24h, category index)."""
    df = df.drop_duplicates().dropna(subset=["card_id", "timestamp", "amount"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["distance_km"] = df.get("distance_km", 0.0)
    df["distance_km"] = df["distance_km"].fillna(0.0)
    df["merchant_category"] = df["merchant_category"].where(df["merchant_category"].isin(CATS), "shopping")
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    ts = df["timestamp"].astype("datetime64[s]").astype("int64").to_numpy()
    t1, t24 = np.zeros(len(df)), np.zeros(len(df))
    for _, idx in df.groupby("card_id").indices.items():
        s = ts[idx]
        pos = np.arange(1, len(s) + 1)
        t1[idx] = pos - np.searchsorted(s, s - 3600, side="right")
        t24[idx] = pos - np.searchsorted(s, s - 86400, side="right")
    df["txn_1h"], df["txn_24h"] = t1, t24
    df["hour"] = df["timestamp"].dt.hour
    df["log_amount"] = np.log1p(df["amount"])
    df["cat_idx"] = df["merchant_category"].map(CAT_IDX).astype(int)
    if "is_fraud" not in df:
        df["is_fraud"] = 0
    return df.sort_values("timestamp").reset_index(drop=True)


def best_threshold(y, s) -> float:
    if y.sum() == 0:
        return float(np.percentile(s, 99))
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.argmax(f[:-1]))])


def metrics(y, s, thr) -> dict:
    pred = (s >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"auprc": round(float(average_precision_score(y, s)), 4),
            "auc_roc": round(float(roc_auc_score(y, s)), 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}


# ---------------------------------------------------------------- the framework
class Engine:
    def __init__(self, weights=(1 / 3, 1 / 3, 1 / 3)):
        self.w, self.alpha, self.threshold = weights, 0.5, 0.5

    # Stage 1 -------------------------------------------------------
    @staticmethod
    def _x(log_amt, hour, dist, t1, t24):
        h = 2 * np.pi * np.asarray(hour, float) / 24
        return np.column_stack([log_amt, np.sin(h), np.cos(h), dist, t1, t24])

    def _raw(self, X):
        Xs = self.scaler.transform(X)
        return {"iforest": -self.iforest.score_samples(Xs),
                "autoencoder": ((self.ae.predict(Xs) - Xs) ** 2).mean(axis=1),
                "lof": -self.lof.score_samples(Xs)}

    def fit(self, normal: pd.DataFrame, seed=0):
        rng = np.random.default_rng(seed)
        X = self._x(normal.log_amount, normal.hour, normal.distance_km, normal.txn_1h, normal.txn_24h)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.iforest = IsolationForest(n_estimators=200, random_state=seed).fit(Xs)
        self.ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=80, random_state=seed).fit(Xs, Xs)
        sub = Xs[rng.choice(len(Xs), min(6000, len(Xs)), replace=False)]
        self.lof = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(sub)
        raw = self._raw(X)   # min-max bounds from training scores (train max as upper bound)
        self.norm = {k: (v.min(), v.max()) for k, v in raw.items()}
        self._profiles(normal)
        bg = normal.sample(min(40, len(normal)), random_state=seed)
        self.background = bg[["amount", "hour", "distance_km", "txn_1h", "txn_24h", "cat_idx"]].to_numpy(float)
        return self

    # Stage 2 -------------------------------------------------------
    def _profiles(self, d):
        cards = sorted(d.card_id.unique())
        self.card_index = {c: i for i, c in enumerate(cards)}
        n = len(cards) + 1                                   # last row = global fallback (cold start)
        self.P_mu, self.P_sd = np.zeros(n), np.zeros(n)
        self.P_dm, self.P_ds = np.zeros(n), np.zeros(n)
        self.P_hour, self.P_cat = np.zeros((n, 24)), np.zeros((n, len(CATS)))

        def fill(i, g):
            self.P_mu[i], self.P_sd[i] = g.log_amount.mean(), g.log_amount.std(ddof=0)
            self.P_dm[i], self.P_ds[i] = g.distance_km.mean(), g.distance_km.std(ddof=0)
            h = np.bincount(g.hour, minlength=24) + 0.5
            c = np.bincount(g.cat_idx, minlength=len(CATS)) + 0.5
            self.P_hour[i], self.P_cat[i] = h / h.sum(), c / c.sum()
        fill(n - 1, d)
        for i in range(n - 1):
            fill(i, d)  # overwritten below for cards with enough history
        for c, g in d.groupby("card_id"):
            if len(g) < 10:
                fill(self.card_index[c], d)
            else:
                fill(self.card_index[c], g)

    def lookup(self, card_ids):
        g = len(self.P_mu) - 1
        return np.array([self.card_index.get(c, g) for c in card_ids])

    def _personal(self, ci, log_amt, hour, dist, cat):
        z = np.abs(log_amt - self.P_mu[ci]) / np.maximum(self.P_sd[ci], 0.25)
        h = np.asarray(hour).astype(int)
        p3 = self.P_hour[ci, (h - 1) % 24] + self.P_hour[ci, h] + self.P_hour[ci, (h + 1) % 24]
        dz = np.maximum(0, (dist - self.P_dm[ci]) / np.maximum(self.P_ds[ci], 1))
        comp = {"amount": np.clip(z / 4, 0, 1), "hour": 1 - np.clip(p3 / 0.25, 0, 1),
                "category": 1 - np.clip(self.P_cat[ci, cat] / 0.2, 0, 1), "distance": np.clip(dz / 5, 0, 1)}
        return 0.4 * comp["amount"] + 0.2 * comp["hour"] + 0.2 * comp["category"] + 0.2 * comp["distance"]

    # combined ------------------------------------------------------
    def score_arrays(self, ci, log_amt, hour, dist, t1, t24, cat, alpha=None):
        a = self.alpha if alpha is None else alpha
        raw = self._raw(self._x(log_amt, hour, dist, t1, t24))
        out = {}
        for k, v in raw.items():
            lo, hi = self.norm[k]
            out[k] = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
        out["ensemble"] = self.w[0] * out["iforest"] + self.w[1] * out["autoencoder"] + self.w[2] * out["lof"]
        out["personal"] = self._personal(ci, log_amt, hour, dist, cat)
        out["final"] = a * out["ensemble"] + (1 - a) * out["personal"]   # Final = α·Ens + β·Personal
        return out

    def score_df(self, df, alpha=None):
        return self.score_arrays(self.lookup(df.card_id), df.log_amount.to_numpy(), df.hour.to_numpy(),
                                 df.distance_km.to_numpy(), df.txn_1h.to_numpy(), df.txn_24h.to_numpy(),
                                 df.cat_idx.to_numpy(), alpha)

    # Stage 3 -------------------------------------------------------
    def explain(self, card_id, vals):
        """vals = [amount, hour, distance_km, txn_1h, txn_24h, cat_idx]. SHAP on the FINAL score."""
        ci = self.lookup([card_id])[0]

        def f(M):
            M = np.atleast_2d(M)
            s = self.score_arrays(np.full(len(M), ci), np.log1p(M[:, 0]), M[:, 1], M[:, 2], M[:, 3], M[:, 4],
                                  np.clip(np.round(M[:, 5]).astype(int), 0, len(CATS) - 1))
            return s["final"]
        x = np.array([vals], float)
        try:
            import shap
            ex = shap.KernelExplainer(f, self.background)
            sv = np.asarray(ex.shap_values(x, nsamples=200)).reshape(-1)
            base = float(np.ravel(ex.expected_value)[0])
        except Exception:   # fallback: marginal replacement against background rows
            base = float(f(self.background).mean())
            sv = np.zeros(6)
            for j in range(6):
                alt = np.repeat(x, len(self.background), 0)
                alt[:, j] = self.background[:, j]
                sv[j] = float(f(x)[0] - f(alt).mean())
        pred = float(f(x)[0])
        i = self.card_index.get(card_id, len(self.P_mu) - 1)
        top_cats = [CATS[k] for k in np.argsort(-self.P_cat[i])[:3]]
        names = {"merchant_category": CATS[int(vals[5])]}
        contrib = [{"feature": EXPL[j], "value": names.get(EXPL[j], round(float(vals[j]), 2)),
                    "shap": round(float(sv[j]), 4)} for j in range(6)]
        contrib.sort(key=lambda c: -abs(c["shap"]))
        reasons = [f"{c['feature']} = {c['value']} raised the risk by {c['shap']:.3f}"
                   for c in contrib if c["shap"] > 0.005][:3]
        return {"base_value": round(base, 4), "score": round(pred, 4), "contributions": contrib,
                "reasons": reasons,
                "card_profile": {"typical_amount": round(float(np.expm1(self.P_mu[i])), 2),
                                 "typical_hour": int(np.argmax(self.P_hour[i])), "top_categories": top_cats,
                                 "typical_distance_km": round(float(self.P_dm[i]), 1)}}


# ---------------------------------------------------------------- pipeline
def run_pipeline(raw_df: pd.DataFrame) -> dict:
    df = prepare(raw_df)
    q60, q75 = df.timestamp.quantile([.60, .75])
    train = df[(df.timestamp <= q60) & (df.is_fraud == 0)]
    val, test = df[(df.timestamp > q60) & (df.timestamp <= q75)], df[df.timestamp > q75].copy()
    eng = Engine().fit(train)

    sv = eng.score_df(val)
    yv = val.is_fraud.to_numpy()
    best, best_ap = 0.5, -1
    for a in np.linspace(0, 1, 11):                       # tune α (β = 1-α) on validation AUPRC
        s = a * sv["ensemble"] + (1 - a) * sv["personal"]
        ap = average_precision_score(yv, s) if yv.sum() else 0
        if ap > best_ap:
            best, best_ap = float(a), ap
    eng.alpha = best

    sv, st = eng.score_df(val), eng.score_df(test)
    yt = test.is_fraud.to_numpy()
    rows = []
    for key, label in [("iforest", "Isolation Forest (alone)"), ("autoencoder", "Autoencoder (alone)"),
                       ("lof", "LOF (alone)"), ("ensemble", "Ensemble (Stage 1)"),
                       ("final", "Ensemble + Cardholder Profile (Stage 1+2)")]:
        thr = best_threshold(yv, sv[key])
        rows.append({"model": label, "key": key, "threshold": round(thr, 4), **metrics(yt, st[key], thr)})
    eng.threshold = next(r["threshold"] for r in rows if r["key"] == "final")
    for k in ["iforest", "autoencoder", "lof", "ensemble", "personal", "final"]:
        test[k] = st[k]
    test["flagged"] = (test["final"] >= eng.threshold).astype(int)
    test = test.reset_index(drop=True)
    return {"engine": eng, "test": test, "metrics": rows, "alpha": round(eng.alpha, 2),
            "summary": {"transactions": len(df), "cards": int(df.card_id.nunique()),
                        "frauds": int(df.is_fraud.sum()), "train_rows": len(train),
                        "test_rows": len(test), "test_frauds": int(yt.sum())}}


def drift_experiment(raw_df: pd.DataFrame, windows=5) -> list:
    """Stage 4: static model (trained once) vs sliding-window retraining, per time window."""
    df = prepare(raw_df)
    t0, t1 = df.timestamp.min(), df.timestamp.max()
    edges = [t0 + (t1 - t0) * f for f in np.linspace(0.3, 1.0, windows + 1)]
    first = df[(df.timestamp <= edges[0]) & (df.is_fraud == 0)]
    static = Engine().fit(first)
    out = []
    for k in range(windows):
        cur = df[(df.timestamp > edges[k]) & (df.timestamp <= edges[k + 1])]
        prev = df[(df.timestamp > (edges[k - 1] if k else t0)) & (df.timestamp <= edges[k]) & (df.is_fraud == 0)]
        if cur.is_fraud.sum() == 0 or len(prev) < 200:
            continue
        sliding = Engine().fit(prev)
        y = cur.is_fraud.to_numpy()
        out.append({"window": k + 1, "frauds": int(y.sum()),
                    "static_auprc": round(float(average_precision_score(y, static.score_df(cur)["final"])), 4),
                    "sliding_auprc": round(float(average_precision_score(y, sliding.score_df(cur)["final"])), 4)})
    return out
