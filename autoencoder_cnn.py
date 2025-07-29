# autoencoder_cnn.py
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # 32x32
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1),    # 256x256
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
