import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# STEP 4: SCALE RFM
# --------------------------------------------------
def scale_rfm(rfm_df, scaler_path):
    features = rfm_df[["Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=["Recency", "Frequency", "Monetary"],
        index=rfm_df.index
    )

    return scaled_df


# --------------------------------------------------
# STEP 5A: FIND BEST K (ANALYSIS ONLY)
# --------------------------------------------------
def find_best_k(scaled_rfm, k_min=2, k_max=8):
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_rfm)
        score = silhouette_score(scaled_rfm, labels)

        print(f"k={k} | silhouette_score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


# --------------------------------------------------
# STEP 5B: TRAIN KMEANS
# --------------------------------------------------
def train_kmeans(scaled_rfm, rfm_df, n_clusters, model_path):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_rfm)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    rfm_clustered = rfm_df.copy()
    rfm_clustered["Cluster"] = clusters

    return rfm_clustered


# --------------------------------------------------
# STEP 6A: CLUSTER SUMMARY
# --------------------------------------------------
def cluster_summary(rfm_clustered):
    summary = (
        rfm_clustered
        .groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .round(2)
        .sort_values("Avg_Monetary", ascending=False)
    )

    return summary


# --------------------------------------------------
# STEP 6B: ASSIGN BUSINESS LABELS
# --------------------------------------------------
def assign_segment_labels(summary):
    labels = {}

    sorted_clusters = summary.sort_values(
        ["Avg_Monetary", "Avg_Frequency", "Avg_Recency"],
        ascending=[False, False, True]
    ).index.tolist()

    labels[sorted_clusters[0]] = "High-Value"
    labels[sorted_clusters[1]] = "Regular"
    labels[sorted_clusters[-1]] = "At-Risk"

    for c in sorted_clusters[2:-1]:
        labels[c] = "Occasional"

    return labels
