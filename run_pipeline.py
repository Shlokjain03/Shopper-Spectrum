from src.load_data import load_data
from src.preprocess import clean_data
from src.rfm import build_rfm
from src.clustering import (
    scale_rfm,
    find_best_k,
    train_kmeans,
    cluster_summary,
    assign_segment_labels
)

import joblib


# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = load_data("data/online_retail.csv")

# -------------------------------
# STEP 2: CLEAN DATA
# -------------------------------
clean_df = clean_data(df)

# -------------------------------
# STEP 3: BUILD RFM
# -------------------------------
rfm_df = build_rfm(clean_df)

# -------------------------------
# STEP 4: SCALE RFM
# -------------------------------
scaled_rfm = scale_rfm(rfm_df, "models/rfm_scaler.pkl")

# -------------------------------
# STEP 5: TRAIN KMEANS
# -------------------------------
best_k = find_best_k(scaled_rfm)
print("Best k selected (by silhouette):", best_k)

# Override silhouette-optimal but useless k=2
if best_k == 2:
    print("Overriding k=2 → using k=4 for meaningful segmentation")
    best_k = 4

rfm_clustered = train_kmeans(
    scaled_rfm,
    rfm_df,
    n_clusters=best_k,
    model_path="models/kmeans_model.pkl"
)

print("\nCluster distribution:")
print(rfm_clustered["Cluster"].value_counts())

# -------------------------------
# STEP 6: INTERPRET & LABEL
# -------------------------------
summary = cluster_summary(rfm_clustered)
print("\nCluster Summary:\n", summary)

labels = assign_segment_labels(summary)
rfm_clustered["Segment"] = rfm_clustered["Cluster"].map(labels)

print("\nFinal Segments:\n")
print(
    rfm_clustered
    .groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
    )
    .round(2)
)

# -------------------------------
# STEP 7: SAVE CLUSTER LABELS
# -------------------------------
joblib.dump(labels, "models/cluster_labels.pkl")
print("\nCluster labels saved to models/cluster_labels.pkl")
