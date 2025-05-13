import torch.nn as nn
from config_i_dane import config


class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        # Warstwy konwolucyjne
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config['nc'], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Warstwy w pełni połączone
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),  # 128x128 -> 64x64 -> 32x32
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Spłaszczenie
        x = self.fc_layers(x)
        return x