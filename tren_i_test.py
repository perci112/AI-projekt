import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import GestureCNN
from config_i_dane import config, prepare_datasets, create_loaders, load_single_image
import os
from datetime import datetime

# Inicjalizacja TensorBoard
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/gesture_recognition_{timestamp}')


def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    # Zapis do TensorBoard
    writer.add_scalar('Training Loss', epoch_loss, epoch)
    writer.add_scalar('Training Accuracy', epoch_acc, epoch)

    print(f'Train Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    return epoch_acc


def test_model(model, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    # Zapis do TensorBoard
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', test_acc, epoch)

    print(f'Test Epoch {epoch}: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    return test_acc


def predict_gesture(model, image_path, class_names):
    """Predykcja dla pojedynczego obrazu"""
    model.eval()
    image_tensor = load_single_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


def save_model(model, path='model_weights.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model, path='model_weights.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model loaded from {path}')
    return model


if __name__ == '__main__':
    # Przygotowanie danych
    train_dataset, test_dataset = prepare_datasets()
    train_loader, test_loader = create_loaders(train_dataset, test_dataset)
    class_names = train_dataset.classes

    # Inicjalizacja modelu
    model = GestureCNN(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Trening
    best_acc = 0.0
    for epoch in range(1, config['num_epochs'] + 1):
        train_acc = train_model(model, train_loader, criterion, optimizer, epoch)
        test_acc = test_model(model, test_loader, criterion, epoch)

        # Zapisz najlepszy model
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, 'best_model.pth')

    writer.close()

    # Przykład predykcji
    test_image = 'test_gesture.jpg'  # Zmień na ścieżkę do swojego obrazu
    if os.path.exists(test_image):
        prediction = predict_gesture(model, test_image, class_names)
        print(f'Predicted gesture: {prediction}')