"""
Coffee Bean Classifier - Model Training Script
Author: Nole Mohammed (Data Science & Performance)

Trains two models:
1. MobileNetV2 (Transfer Learning) - Primary model
2. Improved CNN (Benchmark comparison)
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import os

# ============================================
# 0. CREATE REQUIRED DIRECTORIES
# ============================================

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# ============================================
# 1. DATA PREPROCESSING & AUGMENTATION
# ============================================

train_dir = 'Train'  # Folder containing 'Good' and 'Bad'

# Training data (with augmentation)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    validation_split=0.2
)

# Validation data (NO augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ============================================
# 2. CALLBACKS (SMART TRAINING)
# ============================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ============================================
# 3. MOBILENETV2 MODEL
# ============================================

base_brain = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_brain.trainable = False
print("MobileNetV2 base model loaded and frozen.")

model = tf.keras.Sequential([
    base_brain,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# 4. TRAIN MOBILENETV2
# ============================================

print("\n" + "="*50)
print("Training MobileNetV2 Model...")
print("="*50)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop],
    verbose=1
)

# ============================================
# 5. SAVE ACCURACY PLOT
# ============================================

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Coffee Bean Classifier - Accuracy (MobileNetV2)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plot_path = '../outputs/accuracy_plot_mobilenet.png'
plt.savefig(plot_path)
plt.show()

print(f"Accuracy plot saved to {plot_path}")

# ============================================
# 6. SAVE MODEL
# ============================================

model_path = '../models/mobilenetv2_model.keras'
model.save(model_path)
print(f"MobileNetV2 model saved to {model_path}")

# ============================================
# 7. IMPROVED CNN MODEL (Benchmark)
# ============================================

print("\n" + "="*50)
print("Training Improved CNN Model...")
print("="*50)

simple_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

simple_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

simple_history = simple_model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=[early_stop],
    verbose=1
)

# Save CNN model
cnn_path = '../models/cnn_model.keras'
simple_model.save(cnn_path)
print(f"Simple CNN model saved to {cnn_path}")

# ============================================
# 8. FINAL RESULTS
# ============================================

print("\n" + "="*50)
print("TRAINING COMPLETE - RESULTS SUMMARY")
print("="*50)

mobilenet_acc = history.history['val_accuracy'][-1]
cnn_acc = simple_history.history['val_accuracy'][-1]

print(f"MobileNetV2 - Final Validation Accuracy: {mobilenet_acc:.4f}")
print(f"Simple CNN  - Final Validation Accuracy: {cnn_acc:.4f}")

if mobilenet_acc >= 0.80:
    print("\n✅ Target Achieved: ≥ 80% Accuracy")
else:
    print("\n⚠️ Target not reached yet — consider more data or fine-tuning")
