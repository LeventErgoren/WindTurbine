import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPOCHS = 20
BATCH_SIZE = 32
WINDOW_SIZE = 30
STEP_SIZE = 1

FILE_VERILER = "../gercekVeriler/Ocak-Kasım 10 dk lık veriler.xlsm"
FILE_TEMP1 = "../gercekVeriler/Ocak-Kasım 10 dk lık sıcaklıklar 1.xlsm"
FILE_TEMP2 = "../gercekVeriler/Ocak-Kasım 10 dk lık sıcaklıklar 2.xlsm"

def load_and_merge_data(plant_id, veriler_path, temp1_path, temp2_path):
    """
    Belirtilen Plant ID'sine ait verileri üç farklı excel dosyasından okuyup
    zaman etiketine göre (Time) birleştirir ve sayısal bir matris döndürür.
    """
    print(f"   📂 Plant {plant_id} verileri okunuyor... (Excel boyutundan dolayı 1-2 dk sürebilir)")
    
    df_v = pd.read_excel(veriler_path, sheet_name=0, engine='openpyxl')
    df_v = df_v[df_v['Plant'] == plant_id].copy()
    
    df_t1 = pd.read_excel(temp1_path, sheet_name=0, engine='openpyxl')
    df_t1 = df_t1[df_t1['Plant'] == plant_id].copy()
    
    df_t2 = pd.read_excel(temp2_path, sheet_name=0, engine='openpyxl')
    df_t2 = df_t2[df_t2['Plant'] == plant_id].copy()
    
    df_v['Time'] = pd.to_datetime(df_v['Time'])
    df_t1['Time'] = pd.to_datetime(df_t1['Time'])
    df_t2['Time'] = pd.to_datetime(df_t2['Time'])
    
    merged = pd.merge(df_v, df_t1, on=['Plant', 'Serial no.', 'Time'], how='inner')
    merged = pd.merge(merged, df_t2, on=['Plant', 'Serial no.', 'Time'], how='inner')
    
    merged = merged.sort_values('Time').reset_index(drop=True)
    
    cols_to_drop = ['Plant', 'Serial no.', 'Time', 'Operating hours']
    for c in cols_to_drop:
        if c in merged.columns:
            merged = merged.drop(columns=[c])
            
    merged = merged.select_dtypes(include=[np.number])
    
    merged = merged.ffill().fillna(0)
    
    print(f"      -> Plant {plant_id}: {merged.shape[0]} satır, {merged.shape[1]} sensör verisi başarıyla birleştirildi.")
    return merged.values

def create_sequences(data, window_size, step_size):
    """Veriyi LSTM için [Örnek Sayısı, Zaman Adımı, Özellik Sayısı] formatına çevirir."""
    X, y = [], []
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])  
    return np.array(X), np.array(y)


print("1/5: Sağlam Veriler (Plant 1) Yükleniyor...")
raw_healthy = load_and_merge_data(1, FILE_VERILER, FILE_TEMP1, FILE_TEMP2)
if len(raw_healthy) == 0: print("HATA: Sağlam veri yüklenemedi!"); exit()

print("\n2/5: Hasarlı/Test Verileri (Plant 12) Yükleniyor...")
raw_damaged = load_and_merge_data(12, FILE_VERILER, FILE_TEMP1, FILE_TEMP2)
if len(raw_damaged) == 0: print("HATA: Hasarlı veri yüklenemedi!"); exit()

print("\n3/5: Veriler Ölçekleniyor ve LSTM Dizileri Oluşturuluyor...")
scaler = MinMaxScaler()

scaled_healthy = scaler.fit_transform(raw_healthy)
scaled_damaged = scaler.transform(raw_damaged)

X_healthy, y_healthy = create_sequences(scaled_healthy, WINDOW_SIZE, STEP_SIZE)
X_damaged, y_damaged = create_sequences(scaled_damaged, WINDOW_SIZE, STEP_SIZE)

X_train, X_val, y_train, y_val = train_test_split(X_healthy, y_healthy, test_size=0.1, random_state=42)

print(f"   -> Eğitim Dizileri (Normal): X:{X_train.shape}, y:{y_train.shape}")
print(f"   -> Test Dizileri (Hasarlı): X:{X_damaged.shape}, y:{y_damaged.shape}")

print("\n4/5: LSTM Modeli Kuruluyor...")
input_features = X_train.shape[2] 

model = Sequential([
    LSTM(128, input_shape=(WINDOW_SIZE, input_features)),
    Dropout(0.2),
    Dense(input_features) 
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

print(f"\nEğitim Başlıyor ({EPOCHS} Epoch)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)

print("\n5/5: Sonuçlar ve Anomali Metrikleri Hesaplanıyor...")

train_pred = model.predict(X_train)
train_mae = np.mean(np.abs(train_pred - y_train), axis=1)
THRESHOLD = np.percentile(train_mae, 99.5)

val_pred = model.predict(X_val)
val_mae = np.mean(np.abs(val_pred - y_val), axis=1)

damaged_pred = model.predict(X_damaged)
damaged_mae = np.mean(np.abs(damaged_pred - y_damaged), axis=1)

print("\n--- Train Results ---")
print(f"Threshold : {THRESHOLD:.8f}")
print(f"Train MAE mean:  {np.mean(train_mae):.8f}")
print(f"Test (Damaged) MAE mean:  {np.mean(damaged_mae):.8f}")

anomalies = damaged_mae > THRESHOLD
detected_indices = np.where(anomalies)[0]

df_detected = pd.DataFrame({
    'window_index': detected_indices,
    'mae_loss': damaged_mae[detected_indices],
    'threshold': THRESHOLD,
    'anomaly': True
})

print("\nDetected damaged windows:")
print(df_detected.head(10).to_string())
print(f"...\n[{len(df_detected)} rows x 4 columns]")

y_true_val = np.zeros(len(val_mae))
y_pred_val = (val_mae > THRESHOLD).astype(int)

y_true_damaged = np.ones(len(damaged_mae))
y_pred_damaged = (damaged_mae > THRESHOLD).astype(int)

y_true_all = np.concatenate([y_true_val, y_true_damaged])
y_pred_all = np.concatenate([y_pred_val, y_pred_damaged])

acc = accuracy_score(y_true_all, y_pred_all)
prec = precision_score(y_true_all, y_pred_all)
rec = recall_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all)

print("\n--- Other Results ---")
print(f"Accuracy: {acc:.4f} - The rate of correct predictions.")
print(f"Precision: {prec:.4f} - The rate of how many predictions made on the damage are true.")
print(f"Recall: {rec:.4f} - Missing Value, the model points most damaged values as undamaged.")
print(f"F1 Score: {f1:.4f} - The F1 score is the mean of Precision and Recall.")

cm = confusion_matrix(y_true_all, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Damaged'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Graph')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Validation Graph (MAE)')
plt.legend()
plt.tight_layout()
plt.show()

def plot_prediction_sample(X_seq, y_real, y_pred, title, sample_idx=50, feature_idx=0):
    plt.figure(figsize=(12, 5))
    
    history_data = X_seq[sample_idx, :, feature_idx]
    time_steps = range(len(history_data))
    plt.plot(time_steps, history_data, label='Real Window (History)', color='blue')
    
    plt.plot(WINDOW_SIZE, y_real[sample_idx, feature_idx], marker='o', color='green', label='Real Next Step')
    
    plt.plot(WINDOW_SIZE, y_pred[sample_idx, feature_idx], marker='x', color='red', label='Predicted Next Step')
    
    plt.title(title + f" (Feature: {merged.columns[feature_idx]})")
    plt.xlabel('Timestep')
    plt.ylabel('Scaled Feature Value')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_prediction_sample(X_val, y_val, val_pred, 'Predicted vs Real Next Window (Undamaged Plant 1)', sample_idx=100)
plot_prediction_sample(X_damaged, y_damaged, damaged_pred, 'Predicted vs Real Next Window (Damaged Plant 12)', sample_idx=100)