import pandas as pd


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features for each customer.

    Args:
        df (pd.DataFrame): Cleaned retail dataframe

    Returns:
        pd.DataFrame: RFM dataframe indexed by CustomerID
    """

    # Reference date = last date in dataset + 1 day
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    return rfm
