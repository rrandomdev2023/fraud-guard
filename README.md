# Adaptive Hybrid Ensemble Anomaly Detection for Credit Card Fraud (FastAPI)

Stage 1 Isolation Forest + Autoencoder + LOF ensemble, Stage 2 per-cardholder profile
(Final = α·Ensemble + β·Personal, α tuned on validation), Stage 3 SHAP explanations,
Stage 4 drift experiment (static vs sliding-window retraining).

## Run (Windows, Python 3.10+)
    cd backend
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (API docs).
Click "Train on synthetic data" (about a minute).

## Use real data
Upload a CSV with: card_id, timestamp, amount, merchant_category, distance_km, is_fraud.
To convert Sparkov (kartik2112): card_id=cc_num, timestamp=trans_date_trans_time, amount=amt,
merchant_category=category (mapped into the 10 categories in app/data.py, others become "shopping"),
distance_km=haversine(lat,long,merch_lat,merch_long), is_fraud=is_fraud.
Detectors are fitted only on normal, earlier transactions; evaluation is on the latest 25% of time.

## Notes
- The autoencoder is scikit-learn MLPRegressor (8-3-8 bottleneck) to avoid a TensorFlow install;
  swap in Keras/PyTorch in app/ml.py (Engine.fit) if your guide requires it.
- Metrics use a threshold chosen on the validation window, then reported on the test window.
