import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPOCHS = 20
BATCH_SIZE = 32
NUM_CLASSES = 2 
WINDOW_SIZE = 30  
STEP_SIZE = 1     

FILE_VERILER = "../gercekVeriler/Ocak-Kasım 10 dk lık veriler.xlsm"
FILE_TEMP1 = "../gercekVeriler/Ocak-Kasım 10 dk lık sıcaklıklar 1.xlsm"
FILE_TEMP2 = "../gercekVeriler/Ocak-Kasım 10 dk lık sıcaklıklar 2.xlsm"

def load_data_with_label(plant_id, label, veriler_path, temp1_path, temp2_path):
    """Excel verilerini okur ve belirtilen ETİKETİ (Label) yapıştırır."""
    print(f"   📂 Plant {plant_id} (Etiket: {label}) verileri okunuyor...")
    
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
    
    feats = merged.values
    labels = np.full(len(feats), label)
    
    print(f"      -> {len(feats)} adet veri başarıyla yüklendi.")
    return feats, labels

def create_windows(data, window_size, step_size=1):
    """Zaman serisi verisini pencerelere böler ve Dense katmanlar için düzleştirir."""
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    windows = np.array(windows)
    return windows.reshape(windows.shape[0], -1)


print("1/5: Veriler Yükleniyor...")
X_0_raw, _ = load_data_with_label(1, 0, FILE_VERILER, FILE_TEMP1, FILE_TEMP2)
X_1_raw, _ = load_data_with_label(12, 1, FILE_VERILER, FILE_TEMP1, FILE_TEMP2)

if len(X_0_raw)==0 or len(X_1_raw)==0:
    print("HATA: Veri yüklenemedi!")
    exit()

print("\n2/5: Veri Hazırlığı (Scaling, Windowing & One-Hot)...")
scaler = MinMaxScaler()
scaler.fit(np.vstack([X_0_raw, X_1_raw]))

X_0_scaled = scaler.transform(X_0_raw)
X_1_scaled = scaler.transform(X_1_raw)

X_0_win = create_windows(X_0_scaled, WINDOW_SIZE, STEP_SIZE)
X_1_win = create_windows(X_1_scaled, WINDOW_SIZE, STEP_SIZE)

y_0_win = np.full(len(X_0_win), 0)
y_1_win = np.full(len(X_1_win), 1)

X_all = np.vstack([X_0_win, X_1_win])
y_all = np.concatenate([y_0_win, y_1_win])

print(f"   -> Toplam Pencere Sayısı: {len(X_all)}")
print(f"   -> Sınıf Dağılımı: Sağlam Pencereler={len(X_0_win)}, Hasarlı Pencereler={len(X_1_win)}")

y_categorical = to_categorical(y_all, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_categorical, test_size=0.2, random_state=42, stratify=y_all
)

print("\n3/5: Deep Classifier Modeli Kuruluyor...")
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

x = Dense(512, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x) 

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)

output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

print(f"\n4/5: Eğitim Başlıyor ({EPOCHS} Epoch)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\n5/5: Sonuçlar Hesaplanıyor...")

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
class_names = ['Sağlam', 'Hasarlı']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('CONFUSION MATRIX (Karmaşıklık Matrisi)\nKöşegenler Doğru Tahmindir', fontsize=14)
plt.ylabel('Gerçek Durum', fontsize=12)
plt.xlabel('Modelin Tahmini', fontsize=12)
plt.show()

print("\n--- DETAYLI SINIFLANDIRMA RAPORU ---")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Test Başarısı')
plt.title('Model Başarısı (Accuracy)')
plt.ylabel('Doğruluk Oranı')
plt.xlabel('Epoch')
plt.legend()
plt.show()