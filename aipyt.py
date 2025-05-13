import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import torch.serialization  # Import for safe_globals

# Parametry
DATA_DIR = r"D:\pythonProject2\training"
TEST_DATA_DIR = r"D:\pythonProject2\test"
MODEL_PATH = "model.pth"
ENCODER_PATH = "label_encoder.pkl"
IMG_SIZE = (47, 78)
BATCH_SIZE = 2
EPOCHS_PER_ROUND = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformacje
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# Dataset
class ImageDataset(Dataset):
    def __init__(self, data_dir, label_encoder=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                path = os.path.join(data_dir, filename)
                self.images.append(path)

                match = re.search(r'_(\w+_\w+)\.png$', filename)
                if match:
                    label = match.group(1)
                    self.labels.append(label)
                else:
                    print(f"UWAGA: brak etykiety w nazwie: {filename}")

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * (IMG_SIZE[0] // 4) * (IMG_SIZE[1] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Wyświetlanie przykładowych obrazów
def show_sample_images(dataset, label_encoder, num_samples=12):
    plt.figure(figsize=(15, 8))
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(img.squeeze(0), cmap='gray')
        label_name = label_encoder.inverse_transform([label])[0]
        plt.title(label_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Funkcja testowania modelu
def test_model(model, dataloader, label_encoder, num_samples=12):
    model.eval()
    images, true_labels, pred_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            images.append(imgs.cpu())
            true_labels.append(labels)
            pred_labels.append(preds.cpu())

    images = torch.cat(images)
    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)

    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(images[idx].squeeze(0), cmap='gray')
        true_label = label_encoder.inverse_transform([true_labels[idx].item()])[0]
        pred_label = label_encoder.inverse_transform([pred_labels[idx].item()])[0]
        plt.title(f"Prawda: {true_label}\nPred: {pred_label}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Główna część programu
# ---------------------------

# Wczytanie danych treningowych
print("\nWczytywanie danych treningowych...")
train_dataset = ImageDataset(DATA_DIR, transform=transform)
label_encoder = train_dataset.label_encoder

# Zapisz label_encoder do pliku
joblib.dump(label_encoder, ENCODER_PATH)

print("\nPrzykładowe zdjęcia treningowe:")
show_sample_images(train_dataset, label_encoder)

# Podział na train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inicjalizacja modelu
model = CNN(num_classes=len(label_encoder.classes_)).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print("\nWczytywanie istniejącego modelu...")
    # Ensure safe globals include CNN class definition
    with torch.serialization.safe_globals([CNN]):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("\nTworzenie nowego modelu...")

# Optymalizator i funkcja straty
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()

# Trening
print("\nRozpoczynam trenowanie...")
for epoch in range(EPOCHS_PER_ROUND):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoka {epoch+1}/{EPOCHS_PER_ROUND}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100 * correct/total:.2f}%")

# Zapis modelu (zapisujemy tylko state_dict)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel zapisany jako: {MODEL_PATH}")

# Walidacja
print("\nOcena na danych walidacyjnych...")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Dokładność na walidacji: {100 * correct/total:.2f}%")

# Testowanie
print("\nTestowanie na danych testowych...")
# Wczytaj ten sam encoder
label_encoder = joblib.load(ENCODER_PATH)
test_dataset = ImageDataset(TEST_DATA_DIR, label_encoder=label_encoder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Wyświetlanie przykładowych predykcji na danych testowych:")
test_model(model, test_loader, label_encoder)

# Ostateczna ocena
print("\nOcena końcowa na danych testowych:")
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Dokładność na danych testowych: {100 * correct/total:.2f}%")
