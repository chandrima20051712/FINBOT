import pandas as pd
import seaborn as sns
import json
import os
# IMPORTANT: data_handler.cleanup_data_files is imported in app.py now
from data_handler import setup_and_load_data 
from classification_model import run_classification_task
from regression_model import run_regression_task

# Configuration for display and styling
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 1000)

def main():
    """
    Main function to load data, run both classification and regression tasks,
    and return a consolidated dictionary of all results for the web server.
    """
    print("Starting Intricate Financial Analysis System with Kaggle Data...")
    
    # Initialize results container
    all_results = {}
    
    # 1. Load Data
    try:
        # data_handler is now smart enough to look for files based on the 
        # DATA_HANDLER_PATH environment variable set in app.py
        df_transactions, df_nifty = setup_and_load_data()
    except Exception as e:
        error_msg = f"Failed to load data. Please check file names and accessibility. Error: {e}"
        print(error_msg)
        # Return an error structure that the web server can handle
        return {'status': 'error', 'message': error_msg, 'data': None}

    
    # 2. Execute Classification Task
    print("\n" + "="*75)
    print("EXECUTING CLASSIFICATION TASK")
    print("="*75)
    try:
        # Capture the dictionary output from the model function
        classification_output = run_classification_task(df_transactions)
        
        if classification_output['status'] == 'success':
            all_results['classification'] = classification_output['results']
            print("\nClassification Model (SVC) trained and results collected successfully.")
        else:
            all_results['classification'] = {'error': classification_output['error']}
            print(f"\nError in Classification Task: {classification_output['error']}")
            
    except Exception as e:
        all_results['classification'] = {'error': f"Fatal error during Classification: {e}"}
        print(f"\nFatal error running Classification Task: {e}")


    # 3. Execute Regression Task
    print("\n" + "="*75)
    print("EXECUTING REGRESSION TASK")
    print("="*75)
    try:
        # Capture the dictionary output from the model function
        regression_output = run_regression_task(df_nifty)
        
        if regression_output['status'] == 'success':
            all_results['regression'] = regression_output['results']
            print("\nRegression Model (Random Forest) trained and results collected successfully.")
        else:
            all_results['regression'] = {'error': regression_output['error']}
            print(f"\nError in Regression Task: {regression_output['error']}")
            
    except Exception as e:
        all_results['regression'] = {'error': f"Fatal error during Regression: {e}"}
        print(f"\nFatal error running Regression Task: {e}")

    
    print("\n*** Analysis Complete. Returning consolidated results. ***")
    
    # 4. Return the consolidated results dictionary
    # app.py will convert this dictionary into a JSON response.
    return {'status': 'success', 'data': all_results}


if __name__ == "__main__":
    # This block is for direct execution (local testing outside of Flask)
    # Note: When running via Flask (app.py), this block is ignored.
    results = main()
    if results['status'] == 'success':
        # If running directly, print the results structure
        print(json.dumps(results['data'], indent=4, sort_keys=False))
    else:
        print(f"Error: {results['message']}")