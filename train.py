import os
import glob
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Config
DATA_PATH = "./DATASET_SEGMEN5"
FIXED_LR = 0.0003
EPOCHS = 100
BATCH_SIZE = 32

# Data loading
def load_dataset(base_path):
    X, y = [], []
    labels = {'NoStroke': 0, 'MinorStroke': 1, 'ModerateStroke': 2}
    for label_name, label_idx in labels.items():
        files = glob.glob(f"{base_path}/{label_name}/*.csv")
        for f in files:
            df = pd.read_csv(f)
            if df.shape == (640, 14):
                X.append(df.values)
                y.append(label_idx)
    return np.array(X), np.array(y)

def extract_bands(data):
    N, T, C = data.shape
    X_bands = []
    for i in range(N):
        sample = data[i]
        sample_bands = []
        for ch in range(C):
            sig = sample[:, ch]
            coeffs = pywt.wavedec(sig, 'db4', level=4)
            d = pywt.upcoef('a', coeffs[0], 'db4', level=4, take=T)
            t = pywt.upcoef('d', coeffs[1], 'db4', level=4, take=T)
            m = pywt.upcoef('d', coeffs[2], 'db4', level=3, take=T)
            b = pywt.upcoef('d', coeffs[3], 'db4', level=2, take=T)
            sample_bands.append(np.stack([d, t, m, b], axis=-1))
        X_bands.append(np.array(sample_bands).transpose(1, 0, 2).reshape(T, -1))
    return np.array(X_bands)

X_raw, y_raw = load_dataset(DATA_PATH)
X_multi = extract_bands(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi.reshape(-1, 56)).reshape(X_multi.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)

# Model architecture
class MambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=16, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_inner = d_model * expand
        self.in_proj = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d = layers.Conv1D(filters=self.d_inner, kernel_size=4,
                                     padding='same', groups=self.d_inner)
        self.x_proj = layers.Dense(d_state * 2 + 1, use_bias=False)
        self.out_proj = layers.Dense(d_model, use_bias=False)

    def call(self, x):
        z_x = self.in_proj(x)
        z, x = tf.split(z_x, 2, axis=-1)
        x = self.conv1d(x)
        x = tf.nn.silu(x)
        s = self.x_proj(x)
        x = x * tf.nn.sigmoid(s[:, :, :1])
        x = x * tf.nn.silu(z)
        return self.out_proj(x)

def build_msaf_mamba(input_shape):
    inputs = layers.Input(shape=input_shape)
    d1 = layers.Conv1D(64, 3, dilation_rate=1, padding='same')(inputs)
    d3 = layers.Conv1D(64, 3, dilation_rate=3, padding='same')(inputs)
    d5 = layers.Conv1D(64, 3, dilation_rate=5, padding='same')(inputs)
    fused = layers.Concatenate()([d1, d3, d5])
    fused = layers.Activation('gelu')(layers.BatchNormalization()(fused))
    x = MambaBlock(192)(fused)
    x = layers.Dropout(0.3)(x)
    x = MambaBlock(192)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='gelu')(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    return models.Model(inputs, outputs)

# Per-band evaluation
bands = ['Delta', 'Theta', 'Mu', 'Beta']
band_results, band_histories, band_loss_histories = {}, {}, {}

for i, band in enumerate(bands):
    indices = [ch * 4 + i for ch in range(14)]
    X_train_band = X_train[:, :, indices]
    X_test_band = X_test[:, :, indices]

    m_band = build_msaf_mamba((640, 14))
    m_band.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FIXED_LR),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h_band = m_band.fit(X_train_band, y_train, validation_data=(X_test_band, y_test),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    band_results[band] = {
        'val_accuracy': max(h_band.history['val_accuracy']),
        'val_loss': min(h_band.history['val_loss'])
    }
    band_histories[band] = {'train_accuracy': h_band.history['accuracy'],
                             'val_accuracy': h_band.history['val_accuracy']}
    band_loss_histories[band] = {'train_loss': h_band.history['loss'],
                                  'val_loss': h_band.history['val_loss']}
    print(f"{band}: Val Acc = {band_results[band]['val_accuracy']*100:.2f}%, "
          f"Val Loss = {band_results[band]['val_loss']:.4f}")

# Optimizer comparison
opts_dict = {
    'Adam': tf.keras.optimizers.Adam(learning_rate=FIXED_LR),
    'Adamax': tf.keras.optimizers.Adamax(learning_rate=FIXED_LR),
    'Nadam': tf.keras.optimizers.Nadam(learning_rate=FIXED_LR),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=FIXED_LR),
}

optimizer_results = {}
for name, opt in opts_dict.items():
    m = build_msaf_mamba((640, 56))
    m.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h = m.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    optimizer_results[name] = {
        'accuracy': max(h.history['val_accuracy']),
        'loss': min(h.history['val_loss'])
    }
    print(f"{name}: Val Acc = {optimizer_results[name]['accuracy']*100:.2f}%, "
          f"Val Loss = {optimizer_results[name]['loss']:.4f}")

optimizer_df = pd.DataFrame.from_dict(optimizer_results, orient='index')
optimizer_df.columns = ['Accuracy', 'Loss']
optimizer_df['Accuracy (%)'] = optimizer_df['Accuracy'] * 100
print(optimizer_df.round(4))

# Learning rate sweep
lr_variations = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]
lr_results = []

for lr in lr_variations:
    m_lr = build_msaf_mamba((640, 56))
    m_lr.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h_lr = m_lr.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    max_acc = max(h_lr.history['val_accuracy'])
    min_loss = min(h_lr.history['val_loss'])
    lr_results.append({'Learning Rate': lr, 'Accuracy': max_acc, 'Loss': min_loss})
    print(f"LR {lr}: Val Acc = {max_acc*100:.2f}%, Val Loss = {min_loss:.4f}")

lr_df = pd.DataFrame(lr_results)
print(lr_df.round(4))

best_lr_row = lr_df.loc[lr_df['Accuracy'].idxmax()]
BEST_LR = best_lr_row['Learning Rate']
print(f"\nSelected learning rate for final model: {BEST_LR}")

# Final model
model_final = build_msaf_mamba((640, 56))
model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BEST_LR),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_final = model_final.fit(X_train, y_train, validation_data=(X_test, y_test),
                                 epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

loss_val, acc_val = model_final.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc_val*100:.2f}%")
print(f"Final Test Loss: {loss_val:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].plot(history_final.history['accuracy'], label='Training Accuracy')
ax[0].plot(history_final.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Training vs Validation Accuracy')
ax[0].legend(); ax[0].grid(True, linestyle='--', alpha=0.7)

ax[1].plot(history_final.history['loss'], label='Training Loss')
ax[1].plot(history_final.history['val_loss'], label='Validation Loss')
ax[1].set_title('Training vs Validation Loss')
ax[1].legend(); ax[1].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Evaluation
y_pred = np.argmax(model_final.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NoStroke', 'MinorStroke', 'ModerateStroke'],
            yticklabels=['NoStroke', 'MinorStroke', 'ModerateStroke'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.show()

report_dict = classification_report(y_test, y_pred,
                                     target_names=['NoStroke', 'MinorStroke', 'ModerateStroke'],
                                     output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df.round(4))