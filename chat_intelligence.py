import random

def get_ai_response(prompt, analysis_data):
    print("\n--- Generating AI Chat Response ---")
    try:
        if not analysis_data or "regression_results" not in analysis_data or "classification_results" not in analysis_data:
            print("WARNING: No valid analysis data found.")
            return "Please run the financial analysis first, then ask me questions!"

        regression = analysis_data.get("regression_results", {})
        classification = analysis_data.get("classification_results", {})

        r2 = regression.get("r2_score", "N/A")
        mse = regression.get("mse", "N/A")
        forecast_table = regression.get("forecast_table", [])
        sample_forecast = forecast_table[0] if forecast_table else {}
        signal_today = sample_forecast.get("Signal", "HOLD")

        prompt_lower = prompt.lower()

        # --- BUY/SELL related questions ---
        if "buy" in prompt_lower or "sell" in prompt_lower or "hold" in prompt_lower:
            if signal_today == "BUY":
                return (
                    "Based on the latest market analysis, it may be a **good opportunity to BUY**, "
                    "as short-term signals indicate potential upside momentum. "
                    f"The model’s confidence (R²) is {r2:.4f} with MSE {mse:.2f}."
                )
            elif signal_today == "SELL":
                return (
                    "The analysis suggests a **SELL** stance right now due to downward movement trends. "
                    "You may consider reducing exposure in the short term. "
                    f"(Model fit R² = {r2:.4f}, MSE = {mse:.2f})"
                )
            else:
                return (
                    "The system currently suggests you **HOLD**. "
                    "There is no significant signal for a major buy/sell opportunity just yet. "
                    f"(R² = {r2:.4f})"
                )

        # --- Savings-related questions ---
        if "save" in prompt_lower or "saving" in prompt_lower or "goal" in prompt_lower or "invest" in prompt_lower:
            return (
                "Planning savings is smart! 📊\n"
                "You can calculate your target saving by deciding your investment goal.\n"
                "For example:\n"
                "- If your desired stock entry price is ₹200 and you plan to buy 10 shares in 3 months,\n"
                "- You’d need ₹2,000 saved.\n"
                "Use your expense and income insights from the dashboard to optimize your monthly savings towards this."
            )

        # --- Forecast inquiry ---
        if "forecast" in prompt_lower or "future" in prompt_lower or "prediction" in prompt_lower:
            return (
                f"The forecast model achieved an R² of {r2:.4f}, with a Mean Squared Error of {mse:.2f}. "
                f"Based on trends, the next signal is **{signal_today}**. "
                "The model uses moving averages, RSI, and price momentum to predict this."
            )

        # --- Transaction categorization insights ---
        if "transaction" in prompt_lower or "spending" in prompt_lower or "expense" in prompt_lower:
            return (
                "Your transactions are classified into spending categories such as groceries, utilities, and investments. "
                "You can visualize this via the classification chart on your dashboard for deeper expense analysis."
            )

        # --- General fallback ---
        responses = [
            "Your financial analysis is up-to-date! You can ask about buying, selling, or saving strategies.",
            f"The regression model performed strongly with R²={r2:.4f} and MSE={mse:.2f}.",
            f"Today's market stance: **{signal_today}** according to predictive signals.",
            "I'm here to help you understand your finances — ask about forecasts, trends, or goal planning!"
        ]
        return random.choice(responses)

    except Exception as e:
        print(f"Chat Exception: {e}")
        return "Sorry, something went wrong while processing your request. Try again later."
