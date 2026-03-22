import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ------------------ STYLES ------------------
st.markdown("""
<style>
.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.segment-box {
    background-color: #0f172a;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #22c55e;
}
.hint {
    color: #9ca3af;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("models/rfm_scaler.pkl")
    model = joblib.load("models/kmeans_model.pkl")
    labels = joblib.load("models/cluster_labels.pkl")
    return scaler, model, labels

scaler, model, cluster_labels = load_models()

# ------------------ HEADER ------------------
st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendation using RFM + KMeans")

st.divider()

# ------------------ RFM INPUT ------------------
st.subheader("📊 Customer Behaviour Input")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0,
        max_value=500,
        value=30
    )

with col2:
    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=1,
        max_value=500,
        value=5
    )

with col3:
    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Customer Segment"):
    X = np.array([[recency, frequency, monetary]])
    X_scaled = scaler.transform(X)

    cluster = int(model.predict(X_scaled)[0])
    segment = cluster_labels.get(cluster, "Unknown")

    st.divider()
    st.subheader("📈 Prediction Result")

    m1, m2, m3 = st.columns(3)

    m1.metric("Recency", f"{recency} days")
    m2.metric("Frequency", frequency)
    m3.metric("Monetary Value", f"₹ {monetary:,.0f}")

    st.markdown(
        f"""
        <div class="segment-box">
            <h3>Cluster ID: {cluster}</h3>
            <h2>Segment: {segment}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    explanations = {
        "High-Value": "🔥 Loyal and high-spending customers. Prioritise retention and exclusives.",
        "Regular": "🙂 Consistent shoppers. Upsell and cross-sell opportunities.",
        "Occasional": "🛍️ Infrequent buyers. Engage with promotions.",
        "At-Risk": "⚠️ Long inactive. Re-engagement campaigns needed."
    }

    st.success(explanations.get(segment, "No explanation available."))

# ------------------ RECOMMENDATION ------------------
st.divider()
st.subheader("🧠 Product Recommendation")

st.markdown(
    "<p class='hint'>Enter a product name exactly as it appears in the dataset (case-sensitive).</p>",
    unsafe_allow_html=True
)

product_input = st.text_input(
    "Product Name",
    placeholder="e.g. GREEN VINTAGE SPOT BEAKER"
)

if st.button("🛒 Recommend Products"):
    try:
        from src.recommendation import recommend_products

        recommendations = recommend_products(product_input)

        if len(recommendations) == 0:
            st.warning("No recommendations found. Check product name.")
        else:
            st.markdown("### Recommended Products")
            for item in recommendations[:5]:
                st.markdown(f"- {item}")

    except Exception as e:
        st.error("Recommendation system failed.")
        st.caption(str(e))

# ------------------ FOOTER ------------------
st.divider()
st.caption("Built with RFM Analysis, KMeans Clustering & Streamlit")
