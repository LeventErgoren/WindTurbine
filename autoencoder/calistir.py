import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPOCHS = 20
BATCH_SIZE = 32

FILE_VERILER = "../gercekVeriler/Ocak-Kasım 10 dk lık veriler.xlsm"
FILE_TEMP1 = "../gercekVeriler/Ocak-Kasım 10 dk lık sıcaklıklar 1.xlsm"
FILE_TEMP2 = "../gercekVeriler/Ocak-Kasım 10 dk lık sıcaklıklar 2.xlsm"

def load_and_merge_data(plant_id, veriler_path, temp1_path, temp2_path):
    """Excel dosyalarını okur, Plant ID'ye göre filtreler ve birleştirir."""
    print(f"   📂 Plant {plant_id} verileri okunuyor...")
    
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
    
    print(f"      -> Plant {plant_id}: {merged.shape[0]} satır veri başarıyla birleştirildi.")
    return merged.values, merged.columns


print("1/5: Sağlam Veriler (Plant 1) Yükleniyor...")
raw_healthy, feature_names = load_and_merge_data(1, FILE_VERILER, FILE_TEMP1, FILE_TEMP2)
if len(raw_healthy) == 0: print("HATA: Sağlam veri yüklenemedi!"); exit()

print("\n2/5: Hasarlı/Test Verileri (Plant 12) Yükleniyor...")
raw_damaged, _ = load_and_merge_data(12, FILE_VERILER, FILE_TEMP1, FILE_TEMP2)
if len(raw_damaged) == 0: print("HATA: Hasarlı veri yüklenemedi!"); exit()

print("\n3/5: Veriler Ölçekleniyor...")
scaler = MinMaxScaler()
scaled_healthy = scaler.fit_transform(raw_healthy)
scaled_damaged = scaler.transform(raw_damaged)

X_train, X_val = train_test_split(scaled_healthy, test_size=0.1, random_state=42)
X_damaged = scaled_damaged

print(f"   -> Eğitim Verisi (Normal): {X_train.shape}")
print(f"   -> Test Verisi (Hasarlı): {X_damaged.shape}")

print("\n4/5: Autoencoder Modeli Kuruluyor...")
input_dim = X_train.shape[1] 

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation='relu')(encoded) 

decoded = Dense(128, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded) 

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
autoencoder.summary()

print(f"\nEğitim Başlıyor ({EPOCHS} Epoch)...")
history = autoencoder.fit(
    X_train, X_train, 
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, X_val),
    verbose=1
)

print("\n5/5: Sonuçlar ve Yeniden Yapılandırma (Reconstruction) Metrikleri Hesaplanıyor...")

train_pred = autoencoder.predict(X_train)
train_mae = np.mean(np.abs(train_pred - X_train), axis=1)

val_pred = autoencoder.predict(X_val)
val_mae = np.mean(np.abs(val_pred - X_val), axis=1)

damaged_pred = autoencoder.predict(X_damaged)
damaged_mae = np.mean(np.abs(damaged_pred - X_damaged), axis=1)

THRESHOLD = np.percentile(train_mae, 99.5)

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

print("\nDetected damaged windows (or time steps):")
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
print(f"Accuracy: {acc:.4f} The rate of correct predictions.")
print(f"Precision: {prec:.4f} The rate of how many predictions made on the damage are true.")
print(f"Recall: {rec:.4f} Missing Value, the model points most damaged values as undamaged.")
print(f"F1 Score: {f1:.4f} The F1 score is the mean of Precision and Recall.")

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

def plot_reconstruction_sample(X_real, X_pred, title, sample_idx=50):
    plt.figure(figsize=(14, 5))
    
    plt.plot(X_real[sample_idx], label='Real Sensor Values', color='blue', marker='o', alpha=0.7)
    
    plt.plot(X_pred[sample_idx], label='Reconstructed Values', color='red', marker='x', alpha=0.7)
    
    plt.title(title + f" (Time Step / Row: {sample_idx})")
    plt.xlabel('Sensor Feature Index')
    plt.ylabel('Scaled Value')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_reconstruction_sample(X_val, val_pred, 'Reconstruction vs Real (Undamaged Plant 1)', sample_idx=100)

plot_reconstruction_sample(X_damaged, damaged_pred, 'Reconstruction vs Real (Damaged Plant 12)', sample_idx=100)