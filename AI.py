import os
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Parametry
DATA_DIR = r"D:\pythonProject2\training"
TEST_DATA_DIR = r"D:\pythonProject2\test"
MODEL_PATH = "model.keras"
IMG_SIZE = (47, 78)
BATCH_SIZE = 2
EPOCHS_PER_ROUND = 3

def load_data(data_dir, label_encoder=None):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img)
            images.append(img_array)

            match = re.search(r'_(\w+_\w+)\.png$', filename)
            if match:
                label = match.group(1)
                labels.append(label)
            else:
                print(f"UWAGA: nie udało się znaleźć etykiety dla {filename}")

    images = np.array(images, dtype="float32") / 255.0
    images = np.expand_dims(images, -1)
    labels = np.array(labels)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
    else:
        labels_encoded = label_encoder.transform(labels)

    labels_categorical = to_categorical(labels_encoded)
    return images, labels_categorical, label_encoder

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def show_sample_images(images, labels, label_encoder, num_samples=12):
    plt.figure(figsize=(15, 8))
    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        label_idx = np.argmax(labels[i])
        label_name = label_encoder.inverse_transform([label_idx])[0]
        plt.title(label_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_model(model, images, labels, label_encoder, num_samples=12):
    preds = model.predict(images)

    # Wylosuj indeksy
    indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)

    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        true_idx = np.argmax(labels[idx])
        pred_idx = np.argmax(preds[idx])
        true_label = label_encoder.inverse_transform([true_idx])[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        plt.title(f"Prawda: {true_label}\nPredykcja: {pred_label}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Główna część programu
# ---------------------------

# Wczytaj dane treningowe
print("\nWczytywanie danych treningowych...")
images_train, labels_train, label_encoder = load_data(DATA_DIR)

print("\nPrzykładowe zdjęcia treningowe:")
show_sample_images(images_train, labels_train, label_encoder)

# Podział na zbiór treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size=0.2, random_state=42, shuffle=True)

# Wczytaj istniejący model lub stwórz nowy
if os.path.exists(MODEL_PATH):
    print("\nWczytywanie istniejącego modelu...")
    model = load_model(MODEL_PATH)
else:
    print("\nTworzenie nowego modelu...")
    model = build_model(num_classes=labels_train.shape[1])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trenowanie modelu
print("\nRozpoczynam trenowanie...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PER_ROUND,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stopping]
)

# Zapisz model
model.save(MODEL_PATH)
print(f"\nModel zapisany jako: {MODEL_PATH}")

# Ocena modelu na danych walidacyjnych
print("\nOcena na danych walidacyjnych...")
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Dokładność na danych walidacyjnych: {val_acc * 100:.2f}%")

# Testowanie na danych testowych
print("\nTestowanie na danych testowych...")

test_images, test_labels, _ = load_data(TEST_DATA_DIR, label_encoder)

# Testowanie i pokazanie przykładowych wyników
print("Wyświetlanie przykładowych predykcji na danych testowych:")
test_model(model, test_images, test_labels, label_encoder)

# Ostateczna ocena na danych testowych
print("\nOcena końcowa na danych testowych:")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Dokładność na danych testowych: {test_acc * 100:.2f}%")
