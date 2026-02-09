import pandas as pd
import numpy as np
import io
import re

def setup_and_load_data(transactions_file_stream: io.BytesIO, nifty_file_stream: io.BytesIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads datasets from in-memory CSV streams,
    cleans them aggressively, and returns structured DataFrames.
    """
    print("--- 0. Loading Datasets from In-Memory File Streams ---")

    # --- CSV Loader ---
    def safe_read_csv(stream, name):
        """Load CSV with encoding fallbacks."""
        stream.seek(0)
        try:
            return pd.read_csv(stream)
        except UnicodeDecodeError:
            print(f"WARNING: UTF-8 failed for {name}. Trying 'latin-1'.")
            stream.seek(0)
            return pd.read_csv(stream, encoding='latin-1')
        except Exception as e:
            stream.seek(0)
            print(f"WARNING: CSV read issue for {name}. Trying Python engine fallback.")
            try:
                return pd.read_csv(stream, engine='python')
            except Exception as final_e:
                raise Exception(f"CRITICAL: Failed to read {name}. Details: {final_e}")

    # --- 1. Load Both Datasets ---
    df_transactions = safe_read_csv(transactions_file_stream, "Transactions File")
    df_nifty = safe_read_csv(nifty_file_stream, "NIFTY 50 File")

    # --- 2. Column Normalization ---
    def standardize_columns(df, required_cols_map, critical_cols):
        """Rename and validate columns case-insensitively."""
        df_cols_lower = {col.lower(): col for col in df.columns}
        rename_map = {}

        for expected, std in required_cols_map.items():
            if expected.lower() in df_cols_lower:
                rename_map[df_cols_lower[expected.lower()]] = std
        missing = set(critical_cols) - set(rename_map.values())

        if missing:
            raise Exception(f"Missing critical columns: {', '.join(missing)}")

        df.rename(columns=rename_map, inplace=True)
        return df

    TRANSACTION_COLS = {'Date': 'Date', 'Description': 'Description', 'Category': 'Category', 'Amount': 'Amount'}
    df_transactions = standardize_columns(df_transactions, TRANSACTION_COLS, TRANSACTION_COLS.values())

    NIFTY_COLS = {'Date': 'Date', 'Close': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'}
    CRITICAL_NIFTY = ['Date', 'Close', 'Open', 'High', 'Low']
    df_nifty = standardize_columns(df_nifty, NIFTY_COLS, CRITICAL_NIFTY)

    # --- 3. Transactions Cleaning ---
    if 'Date' in df_transactions.columns and 'Amount' in df_transactions.columns:
        start_rows = len(df_transactions)
        df_transactions['Date'] = df_transactions['Date'].astype(str).str.strip()

        # ✅ FIXED: Correct regex character class ordering to avoid range error
        df_transactions['Date'] = df_transactions['Date'].str.replace(r'[./\\-]', '-', regex=True)

        # Extract the valid date part only
        df_transactions['Date'] = df_transactions['Date'].str.extract(r'^([\d./-]+)', expand=False).fillna(df_transactions['Date'])

        # Parse to datetime (assume Indian DD-MM-YYYY)
        df_transactions['Date'] = pd.to_datetime(df_transactions['Date'], errors='coerce', dayfirst=True)
        df_transactions.dropna(subset=['Date'], inplace=True)

        # Clean Amount field
        df_transactions['Amount'] = df_transactions['Amount'].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
        df_transactions['Amount'] = pd.to_numeric(df_transactions['Amount'], errors='coerce')
        df_transactions.dropna(subset=['Amount'], inplace=True)

        # Feature engineering
        df_transactions['DayOfWeek'] = df_transactions['Date'].dt.dayofweek
        df_transactions['Month'] = df_transactions['Date'].dt.month

        dropped = start_rows - len(df_transactions)
        print(f"Transactions cleaned. Dropped: {dropped}. Remaining: {len(df_transactions)}")

    # --- 4. NIFTY 50 Cleaning ---
    df_nifty['Date'] = df_nifty['Date'].astype(str).str.strip()
    df_nifty['Date'] = df_nifty['Date'].str.replace(r'[./\\-]', '-', regex=True)
    df_nifty['Date'] = pd.to_datetime(df_nifty['Date'], errors='coerce', dayfirst=True)
    df_nifty.dropna(subset=['Date'], inplace=True)
    df_nifty.sort_values(by='Date', inplace=True)

    for col in [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_nifty.columns]:
        df_nifty[col] = df_nifty[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df_nifty[col] = pd.to_numeric(df_nifty[col], errors='coerce')

    # Fill missing Close values
    if 'Close' in df_nifty.columns and df_nifty['Close'].isna().sum() > 0:
        nan_count = df_nifty['Close'].isna().sum()
        df_nifty['Close'] = df_nifty['Close'].interpolate(method='linear')
        print(f"Interpolated {nan_count} NaN values in NIFTY 'Close'.")

    print(f"NIFTY 50 cleaned. Rows: {len(df_nifty)}")

    # --- 5. Relaxed Data Checks ---
    if len(df_transactions) < 10:
        print(f"⚠ WARNING: Only {len(df_transactions)} valid transaction rows found.")
    if len(df_nifty) < 10:
        print(f"⚠ WARNING: Only {len(df_nifty)} NIFTY rows present. Analysis may be limited.")

    # --- 6. Return Clean DataFrames ---
    return df_transactions, df_nifty
