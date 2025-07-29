import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import RGBDImageDataset
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RGB_PATH = '/home/multimediateam/dhiep/image/pcb1/Data/Images/Normal/rgb'
DEPTH_PATH = '/home/multimediateam/dhiep/image/pcb1/Data/Images/Normal/depth'
SAVE_DIR = 'checkpoints'
RESULT_DIR = 'train_vis'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Data
dataset = RGBDImageDataset(RGB_PATH, DEPTH_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = ConvAutoencoder().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Visualization helper
def visualize(input_tensor, recon_tensor, epoch):
    input_tensor = input_tensor.to(DEVICE)
    recon_tensor = recon_tensor.to(DEVICE)
    input_img = input_tensor[:3]  # take RGB only
    recon_img = recon_tensor[:3]
    error = F.l1_loss(input_img, recon_img, reduction='none').mean(0)
    error_map = (error - error.min()) / (error.max() - error.min() + 1e-8)

    input_np = input_img.permute(1,2,0).cpu().numpy()
    recon_np = recon_img.permute(1,2,0).cpu().numpy()
    error_np = error_map.cpu().numpy()
    overlay = input_np.copy()
    overlay[:,:,0] = (overlay[:,:,0] + error_np * 0.8).clip(0,1)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(input_np)
    axs[0].set_title('Input RGB')
    axs[1].imshow(recon_np)
    axs[1].set_title('Reconstruction')
    axs[2].imshow(error_np, cmap='hot')
    axs[2].set_title('Anomaly Map')
    axs[3].imshow(overlay)
    axs[3].set_title('Overlay')
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'epoch_{epoch:03d}.png'))
    plt.close()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(DEVICE)
        # Ensure targets have 4 channels by adding a dummy channel
        targets = torch.cat((targets, torch.zeros_like(targets[:, :1, :, :])), dim=1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

    # visualize the first sample of batch 0 after each epoch
    model.eval()
    with torch.no_grad():
        first_sample, _ = dataset[0]
        input_tensor = first_sample.unsqueeze(0).to(DEVICE)
        recon_tensor = model(input_tensor)
        visualize(first_sample, recon_tensor[0], epoch + 1)

# Save model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cae_rgbd.pth'))
