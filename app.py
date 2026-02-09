import io
import traceback
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

try:
    from data_handler import setup_and_load_data
    from classification_model import run_classification_task
    from regression_model import run_regression_task
    from chat_intelligence import get_ai_response
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure all required modules are in the same directory: data_handler.py, classification_model.py, regression_model.py, chat_intelligence.py.")
    raise

app = Flask(__name__)
CORS(app)

global_analysis_results = {}

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global global_analysis_results
    global_analysis_results = {}

    print("--- Received POST request to /analyze ---")

    transactions_file = request.files.get('transactions_file')
    nifty_file = request.files.get('nifty_file')

    if not transactions_file or not nifty_file:
        missing = []
        if not transactions_file:
            missing.append("Transactions File")
        if not nifty_file:
            missing.append("NIFTY 50 File")
        msg = f"Missing required files: {', '.join(missing)}"
        print(f"ERROR: {msg}")
        return jsonify({"status": "error", "message": msg}), 400

    try:
        df_transactions, df_nifty = setup_and_load_data(
            io.BytesIO(transactions_file.read()), io.BytesIO(nifty_file.read())
        )
        print("Data loaded and cleaned successfully.")
    except Exception as e:
        msg = f"Data loading failed: {e}"
        print(msg)
        traceback.print_exc(file=sys.stdout)
        return jsonify({"status": "error", "message": msg}), 400

    # Classification Task
    try:
        print("=== Running Transaction Classification Task ===")
        classification_results = run_classification_task(df_transactions)
        print("Transaction Classification completed successfully.")
    except Exception as e:
        msg = f"Classification failed: {e}"
        traceback.print_exc(file=sys.stdout)
        return jsonify({"status": "error", "message": msg}), 500

    # Regression Task
    try:
        print("=== Running NIFTY Regression Forecasting Task ===")
        regression_results = run_regression_task(df_nifty)
        print("NIFTY Regression completed successfully.")
    except Exception as e:
        msg = f"Regression failed: {e}"
        traceback.print_exc(file=sys.stdout)
        return jsonify({"status": "error", "message": msg}), 500

    if isinstance(regression_results.get("forecast_table"), pd.DataFrame):
        regression_results["forecast_table"] = regression_results["forecast_table"].to_dict(orient="records")

    global_analysis_results = {
        "regression_results": regression_results,
        "classification_results": classification_results
    }

    print("--- Analysis Complete ---")
    return jsonify({
        "status": "success",
        "message": "Financial analysis completed successfully.",
        "data": global_analysis_results
    }), 200

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    print("--- Received POST request to /chat ---")

    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({"status": "error", "message": "No prompt provided."}), 400

        if not global_analysis_results:
            return jsonify({"status": "error", "message": "Run financial analysis first."}), 400

        ai_response = get_ai_response(prompt, global_analysis_results)
        return jsonify({"status": "success", "response": ai_response}), 200

    except Exception as e:
        msg = f"Chat request failed: {e}"
        traceback.print_exc(file=sys.stdout)
        return jsonify({"status": "error", "message": msg}), 500

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
