from src.predict import load_models, predict_customer_segment

# Load trained artifacts
scaler, model, labels = load_models()

# ---- TEST CASES ----

test_cases = [
    # (Recency, Frequency, Monetary)
    (5, 80, 100000),     # should be High-Value
    (20, 20, 12000),     # should be Regular
    (50, 3, 1500),       # should be Occasional
    (300, 1, 200),       # should be At-Risk
]

for r, f, m in test_cases:
    cluster, segment = predict_customer_segment(
        recency=r,
        frequency=f,
        monetary=m,
        scaler=scaler,
        model=model,
        cluster_labels=labels
    )
    print(f"Input: R={r}, F={f}, M={m} → Cluster {cluster}, Segment: {segment}")
