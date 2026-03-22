import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:

    """
    Load the online retail dataset from a CSV file.
    
    Args:  file_path (str): Path to the CSV file
    
    #Returns:  pd.DataFrame: Loaded dataset with parsed dates
    
    """
        
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=["InvoiceDate"],
            encoding="ISO-8859-1"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    # Basic sanity checks
    required_columns = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }

    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df
