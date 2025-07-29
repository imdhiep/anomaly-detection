import torch
from dataset import RGBDImageDataset
from autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = RGBDImageDataset('.../Anomaly/rgb', '.../Anomaly/depth', transform_size=(128,128))
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load('checkpoints/convAE.pth', map_location=DEVICE))
model.eval()

for i, (x, rgb) in enumerate(loader):
    x = x.to(DEVICE); rgb = rgb.to(DEVICE)
    out = model(x)
    img = x[0].cpu(); rec = out[0].cpu()
    error = torch.abs(rec[:3] - img[:3]).mean(dim=0).numpy()

    fig, axes = plt.subplots(1,4,figsize=(12,3))
    axes[0].imshow(img[:3].permute(1,2,0)); axes[0].set_title('Input RGB')
    axes[1].imshow(rec[:3].permute(1,2,0)); axes[1].set_title('Reconstructed')
    axes[2].imshow(error, cmap='hot'); axes[2].set_title('Anomaly Map')
    overlay = img[:3].permute(1,2,0).numpy()*0.5 + error[...,None]*0.5
    axes[3].imshow(overlay); axes[3].set_title('Overlay')
    for ax in axes: ax.axis('off')
    plt.show()
    if i >= 10: break
