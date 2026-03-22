import joblib
import pandas as pd


def load_models(
    scaler_path="models/rfm_scaler.pkl",
    model_path="models/kmeans_model.pkl",
    labels_path="models/cluster_labels.pkl"
):
    """
    Load trained scaler, KMeans model, and cluster-to-segment mapping.
    Used ONLY for inference (Streamlit or testing).
    """
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    cluster_labels = joblib.load(labels_path)

    return scaler, model, cluster_labels


def predict_customer_segment(
    recency,
    frequency,
    monetary,
    scaler,
    model,
    cluster_labels
):
    """
    Predict customer cluster and business segment using trained artifacts.
    """

    # Create input with correct feature names
    input_df = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )

    # Scale input
    scaled_values = scaler.transform(input_df)

    # Preserve feature names to avoid sklearn warning
    scaled_input = pd.DataFrame(
        scaled_values,
        columns=["Recency", "Frequency", "Monetary"]
    )

    # Predict cluster
    cluster = int(model.predict(scaled_input)[0])

    # Map to business segment
    segment = cluster_labels.get(cluster, "Unknown")

    return cluster, segment
