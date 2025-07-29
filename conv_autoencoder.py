import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=6, latent_dim=128):
        super(ConvAutoencoder, self).__init__()

        # === Encoder ===
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # (6,256,256) -> (32,128,128)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> (64,64,64)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> (128,32,32)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # -> (256,16,16)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # -> (512,8,8)
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # === Bottleneck ===
        self.bottleneck = nn.Sequential(
            nn.Flatten(),  # -> (512*8*8 = 32768)
            nn.Linear(512 * 8 * 8, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8))
        )

        # === Decoder ===
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # -> (256,16,16)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # -> (128,32,32)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # -> (64,64,64)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> (32,128,128)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),  # -> (6,256,256)
            nn.Sigmoid()  # đầu ra ảnh [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
