import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use("Agg")

def plot_graph(y_true, y_pred, title="Actual vs Predicted"):
    """Creates a base64 matplotlib chart comparing actual vs predicted."""
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(y_true, label='Actual Close', color='blue', linewidth=2)
    ax.plot(y_pred, label='Predicted Close', color='orange', linestyle='--', linewidth=2)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Close Price")
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_rsi(df, window=14):
    """Calculates RSI."""
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def run_regression_task(df_prices: pd.DataFrame, buy_threshold=0.001, sell_threshold=-0.001):
    print("\n=== Predict Future Price: Linear Regression Approach ===")
    df = df_prices.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # --- Feature Engineering ---
    if 'Close' not in df.columns:
        raise KeyError("Expected column 'Close' not found in NIFTY CSV file.")

    if 'Open' not in df.columns: df['Open'] = df['Close']
    if 'High' not in df.columns: df['High'] = df['Close']
    if 'Low' not in df.columns: df['Low'] = df['Close']
    if 'Volume' not in df.columns: 
        print("WARNING: 'Volume' column missing, filling with zeros.")
        df['Volume'] = 0

    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['RSI_14'] = calculate_rsi(df, 14)
    df['VOL_10'] = df['Close'].rolling(10).std()
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Target_Close'] = df['Close'].shift(-1)
    df = df.dropna()

    feature_candidates = [
        'Close','Open','High','Low','Volume',
        'MA_5','MA_20','RSI_14','VOL_10','Momentum_5'
    ]
    features = [f for f in feature_candidates if f in df.columns and df[f].notna().all()]

    print(f"Using features: {features}")
    X = df[features]
    y = df['Target_Close']

    # Train-test split
    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    close_today = df.iloc[split:]['Close']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pct_change = (y_pred - close_today) / close_today
    signal = np.where(pct_change > buy_threshold, "BUY",
              np.where(pct_change < sell_threshold, "SELL", "HOLD"))

    forecast_table = pd.DataFrame({
        "Date": df.iloc[split:]['Date'].dt.strftime('%Y-%m-%d'),
        "Actual Close": y_test.round(2),
        "Predicted Close": y_pred.round(2),
        "Percent Change": pct_change.round(4),
        "Signal": signal
    }).head(10)

    regression_graph = plot_graph(y_test, y_pred)
    print(f"Test MSE: {mse:.4f} | R²: {r2:.4f}")

    return {
        "status": "success",
        "mse": mse,
        "r2_score": r2,
        "forecast_table": forecast_table.to_dict(orient="records"),
        "regression_graph": regression_graph
    }
