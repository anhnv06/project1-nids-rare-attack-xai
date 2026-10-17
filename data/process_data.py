import os
import zipfile
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE

# Thiết lập đường dẫn thư mục
current_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(current_dir, "raw")
processed_dir = os.path.join(current_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

# 1. Tìm hoặc giải nén file CSV trong data/raw
csv_path = None
for root, _, files in os.walk(raw_dir):
    for f in files:
        if f.endswith(".csv"):
            csv_path = os.path.join(root, f)
            break
    if csv_path:
        break

if not csv_path:
    for f in os.listdir(raw_dir):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(raw_dir, f), "r") as z:
                z.extractall(raw_dir)
            break
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break

print("Đường dẫn CSV:", csv_path)

# 2. Đọc dữ liệu và lọc 8 cột
drop_cols = [
    'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR',
    'L4_SRC_PORT',
    'L4_DST_PORT',
    'MIN_TTL',
    'MAX_TTL',
    'DNS_TTL_ANSWER',
    'Label'
]

df = pd.read_csv(csv_path)
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

X = df.drop(columns=['Attack']).astype(np.float32)
y = df['Attack']
feature_names = X.columns.tolist()

print("Số đặc trưng:", X.shape[1])
print("Số lớp:", y.nunique())

# 3. Chia tập Train/Test theo tỷ lệ 70/30 (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# 4. Chuẩn hóa MinMaxScaler (chỉ fit và transform trên Train)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, os.path.join(processed_dir, 'minmax_scaler.pkl'))
print("Đã lưu minmax_scaler.pkl")

# 5. Cân bằng dữ liệu (RandomUnderSampler + Borderline-SMOTE1)
rus = RandomUnderSampler(
    sampling_strategy={'Benign': 100000},
    random_state=42
)
X_under, y_under = rus.fit_resample(X_train_scaled, y_train)

counts = Counter(y_under)
sampling_strategy = {
    cls: 5000
    for cls, count in counts.items()
    if count < 5000
}

smote = BorderlineSMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=5,
    m_neighbors=10,
    kind='borderline-1'
)
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

# 6. Xuất các file kết quả vào data/processed
train_smote = pd.DataFrame(X_resampled, columns=feature_names)
train_smote['Attack'] = np.asarray(y_resampled)
train_smote.to_csv(os.path.join(processed_dir, 'train_smote.csv'), index=False)

# File test giữ nguyên giá trị gốc chưa scale
test_raw = pd.DataFrame(X_test, columns=feature_names)
test_raw['Attack'] = np.asarray(y_test)
test_raw.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)

print("Train sau SMOTE:", train_smote.shape)
print("Test chưa scale:", test_raw.shape)
print("Hoàn tất xử lý dữ liệu.")