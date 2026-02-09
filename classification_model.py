import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")

def save_plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_classification_task(df_transactions: pd.DataFrame):
    print("\n=== EXECUTING CLASSIFICATION TASK (Transaction Categorization) ===")

    df = df_transactions.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Dynamically detect usable columns
    possible_text_cols = ["description", "details", "narration", "remarks", "info"]
    possible_label_cols = ["category", "type", "label", "group"]

    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    label_col = next((col for col in possible_label_cols if col in df.columns), None)

    if not text_col or not label_col:
        print("ERROR: Could not detect description/category columns.")
        raise KeyError(
            f"Expected description/text and category/label columns in data. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    print(f"Final data size for model: {len(df)} samples across {len(y.unique())} categories.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # --- SVM Model Pipeline ---
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', SVC(kernel='linear', probability=True))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # --- Evaluate ---
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True)

    classes = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Transaction Classification - Confusion Matrix")

    cm_b64 = save_plot_to_base64(fig)
    plt.close(fig)

    print("--- Classification Complete ---")

    return {
        "status": "success",
        "accuracy": acc,
        "confusion_matrix_plot": cm_b64,
        "classification_report": report
    }
