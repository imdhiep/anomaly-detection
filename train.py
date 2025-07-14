# train.py - Anomaly Map with Overlay and Save
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from mvtec_loader import MVTecDataset
from autoencoder import Autoencoder
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

# === Cài đặt tham số ===
data_dir = "image/wood"
batch_size = 32
epochs = 20
learning_rate = 1e-3
threshold = 0.95

input_dim = 3 * 256 * 256
hidden_dims = [1024, 512, 256]
latent_dim = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Tải dữ liệu ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = MVTecDataset(data_dir, phase="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MVTecDataset(data_dir, phase="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === Khởi tạo Autoencoder dạng fully connected ===
model = Autoencoder(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    latent_dim=latent_dim,
    activation='relu',
    dropout_rate=0.1,
    batch_norm=True
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# === Train ===
print("\n--- Training Autoencoder ---")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        inputs = images.view(images.size(0), -1)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# === Lưu model ===
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/autoencoder_mvtec.pth")
print("\n✅ Đã lưu model tại checkpoints/autoencoder_mvtec.pth")

# === Đánh giá và hiển thị anomaly map + overlay ===
os.makedirs("outputs", exist_ok=True)
print("\n--- Evaluate on Test Data with Anomaly Map + Overlay ---")
model.eval()
all_losses = []
all_labels = []

with torch.no_grad():
    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        inputs = img.view(img.size(0), -1)
        recon = model(inputs)
        loss_pixelwise = torch.abs(recon - inputs).view(1, 3, 256, 256).squeeze().cpu().numpy()
        anomaly_map = np.mean(loss_pixelwise, axis=0)
        score = anomaly_map.mean()

        all_losses.append(score)
        all_labels.append(label.item())

        if i < 5:
            img_np = img.squeeze().cpu().permute(1, 2, 0).numpy()
            recon_np = recon.view(1, 3, 256, 256).squeeze().cpu().permute(1, 2, 0).numpy()

            # Overlay anomaly map lên ảnh gốc
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            anomaly_map_color = plt.cm.inferno(anomaly_map_norm)[:, :, :3]
            overlay = (0.6 * img_np + 0.4 * anomaly_map_color)
            overlay = np.clip(overlay, 0, 1)

            # Hiển thị
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.title("Original")
            plt.imshow(img_np)
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.title("Reconstruction")
            plt.imshow(recon_np)
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.title("Anomaly Map")
            plt.imshow(anomaly_map, cmap='inferno')
            plt.colorbar()
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.title("Overlay")
            plt.imshow(overlay)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"outputs/sample_{i+1}_result.png")
            plt.show()

# === Tính AUROC ===
y_true = np.array(all_labels)
y_score = np.array(all_losses)
auroc = roc_auc_score(y_true, y_score)
print(f"\n✅ AUROC: {auroc:.4f}")

# === Gợi ý threshold ===
threshold = np.percentile(y_score[y_true == 0], 95)
print(f"Threshold gợi ý: {threshold:.6f}")

# === Dự đoán và thống kê ===
y_pred = (y_score > threshold).astype(int)
TP = np.sum((y_pred == 1) & (y_true == 1))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
