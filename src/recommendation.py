# src/recommendation.py

import joblib
import pandas as pd
import os


MODEL_DIR = "models"
SIMILARITY_PATH = os.path.join(MODEL_DIR, "product_similarity.pkl")
PRODUCT_INDEX_PATH = os.path.join(MODEL_DIR, "product_index.csv")


def recommend_products(product_name: str, top_n: int = 5):
    """
    Recommend similar products based on cosine similarity.

    Args:
        product_name (str): Exact product name from dataset
        top_n (int): Number of recommendations

    Returns:
        list[str]: Recommended product names
    """

    if not os.path.exists(SIMILARITY_PATH):
        raise FileNotFoundError("product_similarity.pkl not found")

    if not os.path.exists(PRODUCT_INDEX_PATH):
        raise FileNotFoundError("product_index.csv not found")

    similarity_df = joblib.load(SIMILARITY_PATH)
    product_index = pd.read_csv(PRODUCT_INDEX_PATH)

    if product_name not in similarity_df.index:
        return []

    scores = similarity_df.loc[product_name]
    scores = scores.sort_values(ascending=False)

    recommendations = scores.iloc[1 : top_n + 1].index.tolist()
    return recommendations
