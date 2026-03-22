import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the online retail dataset.
    Steps:
    - Remove missing CustomerID
    - Remove cancelled invoices
    - Remove non-positive quantity and price
    - Create TotalPrice column
    Args:
        df (pd.DataFrame): Raw retail dataframe
    Returns:
        pd.DataFrame: Cleaned dataframe
    """

# 1. Remove rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])

# Convert CustomerID to integer (safe now)
    df["CustomerID"] = df["CustomerID"].astype(int)

# 2. Remove cancelled invoices (InvoiceNo starts with 'C')
    df = df[~df["InvoiceNo"].str.startswith("C")]

# 3. Remove zero or negative quantities
    df = df[df["Quantity"] > 0]

# 4. Remove zero or negative prices
    df = df[df["UnitPrice"] > 0]

# 5. Create total price column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df
