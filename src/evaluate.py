import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def evaluate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

def plot_dual_confusion_matrix(y_true, y_pred_base, y_pred_smote, classes):
    results = [
        ("base", y_pred_base, "Blues", "confusion_matrix_baseline.png"),
        ("smote", y_pred_smote, "Greens", "confusion_matrix_smote.png")
    ]

    for title, y_pred, cmap, filename in results:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)), normalize="true")
        cm_percent = cm * 100
        annot = np.array([[f"{v:.1f}%" if v >= 0.05 else "" for v in row] for row in cm_percent])

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(cm_percent, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=100,
                    square=True, linewidths=0.5, linecolor="white",
                    xticklabels=classes, yticklabels=classes, annot_kws={"size": 9}, ax=ax)

        ax.set_title(title, fontsize=15, pad=15)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.tick_params(axis="x", labelrotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Đã lưu Confusion Matrix: {save_path}")

def main():

    # 1. tải scaler, label encoder và 2 models
    print("\nđang tải models...")
    scaler = joblib.load(os.path.join(PROCESSED_DIR, "minmax_scaler.pkl"))
    le = joblib.load(os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl"))
    model_base = joblib.load(os.path.join(SAVED_MODELS_DIR, "model_baseline.pkl"))
    model_smote = joblib.load(os.path.join(SAVED_MODELS_DIR, "model_smote.pkl"))
    class_names = le.classes_

    # 2. test
    print("đang tải test...")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    feature_names = [c for c in test_df.columns if c != "Attack"]
    
    X_test_raw = test_df[feature_names].astype(np.float32)
    y_test = le.transform(test_df["Attack"])

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_names
    )

    # 3. dự đoán
    print("\nđang chạy...")
    t0 = time.time()
    y_pred_base = model_base.predict(X_test_scaled)
    y_pred_smote = model_smote.predict(X_test_scaled)
    print(f" thời gian chạy: {time.time() - t0:.2f}s")

    # 4. In kết quả và Vẽ biểu đồ
    metrics_base = evaluate_metrics(y_test, y_pred_base)
    metrics_smote = evaluate_metrics(y_test, y_pred_smote)

    comparison_df = pd.DataFrame([metrics_base, metrics_smote], index=["baseline", "smote"])

    print("\n" + "=" * 60)
    print("so sánh kết quả:")
    print("=" * 60)
    print(comparison_df.map(lambda x: f"{x * 100:.4f}%"))
    print("=" * 60)

    plot_dual_confusion_matrix(y_test, y_pred_base, y_pred_smote, class_names)

if __name__ == "__main__":
    main()