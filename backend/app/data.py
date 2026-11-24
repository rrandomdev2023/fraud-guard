"""Synthetic Sparkov-style transaction generator (used when no CSV is uploaded)."""
import numpy as np
import pandas as pd

CATS = ["grocery", "fuel", "restaurant", "travel", "shopping", "entertainment",
        "health", "electronics", "jewelry", "gift_cards"]
COMMON, RARE = CATS[:7], CATS[7:]
DRIFT_START = 0.8  # fraction of the timeline after which the fraud pattern changes


def generate(n_cards=200, per_card=120, fraud_rate=0.012, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, days = pd.Timestamp("2025-01-01"), 180
    parts = []
    for c in range(n_cards):
        mu, sd = rng.uniform(2.8, 5.0), rng.uniform(0.3, 0.7)   # log-amount profile
        home_hour, home_r = rng.integers(9, 20), rng.uniform(2, 25)
        fav = rng.choice(COMMON, size=3, replace=False)
        n = max(int(rng.normal(per_card, 20)), 40)
        frac = np.sort(rng.uniform(0, 1, n))
        hour = np.clip(np.round(rng.normal(home_hour, 3, n)), 0, 23).astype(int)
        amount = np.exp(rng.normal(mu, sd, n))
        cat = np.where(rng.random(n) < 0.85, rng.choice(fav, n), rng.choice(COMMON, n)).astype(object)
        dist = np.abs(rng.normal(0, home_r / 2, n))
        fraud = rng.random(n) < fraud_rate
        for i in np.where(fraud)[0]:
            drift = frac[i] > DRIFT_START
            kind = rng.choice(["A", "B", "C"], p=[.1, .3, .6] if drift else [.3, .7, 0])
            if kind == "A":     # classic: huge amount, night, far away, rare category
                amount[i] = np.exp(mu) * rng.uniform(4, 12)
                hour[i] = rng.integers(0, 6) if rng.random() < .6 else hour[i]
                cat[i] = rng.choice(RARE); dist[i] = rng.uniform(150, 3000)
            elif kind == "B":   # subtle: only unusual for THIS card
                amount[i] = np.exp(mu + rng.uniform(2.2, 3.2) * sd)
                cat[i] = rng.choice(RARE) if rng.random() < .6 else cat[i]
                dist[i] = rng.uniform(0, home_r * 1.5)
            else:               # drifted: small card-testing style amounts
                amount[i] = np.exp(mu) * rng.uniform(.3, 1.0)
                hour[i] = rng.integers(0, 6); cat[i] = "gift_cards"; dist[i] = rng.uniform(0, 80)
        ts = start + pd.to_timedelta(np.floor(frac * days), unit="D") \
            + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
        parts.append(pd.DataFrame({"card_id": f"C{c:04d}", "timestamp": ts, "amount": amount.round(2),
                                   "merchant_category": cat, "distance_km": dist.round(1),
                                   "is_fraud": fraud.astype(int)}))
    return pd.concat(parts, ignore_index=True)
