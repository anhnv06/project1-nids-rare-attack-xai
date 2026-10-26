import os
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def main():

    # 1. tải scaler
    scaler_path = os.path.join(PROCESSED_DIR, "minmax_scaler.pkl")
    scaler = joblib.load(scaler_path)
    le = LabelEncoder()

    # 2. nạp train
    print("\n[1/2]đang nạp train baseline...")
    train_base_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    feature_names = [c for c in train_base_df.columns if c != "Attack"]

    X_train_base_raw = train_base_df[feature_names].astype(np.float32)
    y_train_base = le.fit_transform(train_base_df["Attack"])
    
    # lưu label encoder
    joblib.dump(le, os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl"))

    # giữ lại tên cột sau scaler
    X_train_base_scaled = pd.DataFrame(
        scaler.transform(X_train_base_raw),
        columns=feature_names
    )

    # 3. nạp train smote
    print("[2/2]đang nạp train smote...")
    train_smote_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_smote.csv"))
    X_train_smote = train_smote_df[feature_names].astype(np.float32)
    y_train_smote = le.transform(train_smote_df["Attack"])

    # cấu hình lightgbm
    lgbm_params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "learning_rate": 0.1,
        "n_estimators": 150,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }

    # 4. train baseline
    print("\ntrain baseline...")
    t0 = time.time()
    model_base = lgb.LGBMClassifier(**lgbm_params)
    model_base.fit(X_train_base_scaled, y_train_base)
    joblib.dump(model_base, os.path.join(SAVED_MODELS_DIR, "model_baseline.pkl"))
    print(f" đã lưu model_baseline.pkl (thời gian: {time.time() - t0:.2f}s)")

    # 5. Huấn luyện và Lưu mô hình SMOTE
    print("\ntrain smote...")
    t1 = time.time()
    model_smote = lgb.LGBMClassifier(**lgbm_params)
    model_smote.fit(X_train_smote, y_train_smote)
    joblib.dump(model_smote, os.path.join(SAVED_MODELS_DIR, "model_smote.pkl"))
    print(f" đã lưu model_smote.pkl (thời gian: {time.time() - t1:.2f}s)")

if __name__ == "__main__":
    main()