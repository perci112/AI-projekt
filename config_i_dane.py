import os
import torch
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader

# Hiperparametry
config = {
    'nc': 3,                # liczba kanałów (RGB)
    'batch_size': 32,       # rozmiar batcha
    'num_epochs': 20,       # liczba epok
    'lr': 0.001,            # learning rate
    'random_seed': 42,      # ziarno losowości
    'num_workers': 4,       # liczba wątków
    'image_size': (128, 128) # rozmiar obrazu
}

# Ustawienie ziarna dla reprodukowalności
torch.manual_seed(config['random_seed'])

# Transformacje obrazów
transform = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

def prepare_datasets(data_dir='dataset'):
    """Przygotowanie datasetów treningowego i testowego"""
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )
    return train_dataset, test_dataset

def create_loaders(train_dataset, test_dataset):
    """Tworzenie DataLoaderów"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    return train_loader, test_loader

def load_single_image(image_path):
    """Ładowanie pojedynczego obrazu do predykcji"""
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Dodanie wymiaru batcha