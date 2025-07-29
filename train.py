import torch, os
from dataset import RGBDImageDataset
from torch.utils.data import DataLoader
from model import EasyNet
import torch.nn.functional as F
import torch.optim as optim

def train():
    rgb_n='/home/multimediateam/dhiep/image/pcb2/Data/Images/Normal/rgb'
    depth_n='/home/multimediateam/dhiep/image/pcb2/Data/Images/Normal/depth'
    rgb_a='/home/multimediateam/dhiep/image/pcb2/Data/Images/Anomaly/rgb'
    depth_a='/home/multimediateam/dhiep/image/pcb2/Data/Images/Anomaly/depth'

    ds = RGBDImageDataset(rgb_n, depth_n, rgb_a, depth_a, transform_size=(128,128))
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EasyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for inp, label in loader:
            inp = inp.to(device)
            recon, amap = model(inp)

            recon = F.interpolate(recon, size=inp.shape[2:], mode='bilinear', align_corners=False)

            loss_rgb = F.mse_loss(recon[:, :3], inp[:, :3])
            loss_depth = F.mse_loss(recon[:, 3:4], inp[:, 3:4])
            loss = loss_rgb + loss_depth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/50, Loss {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), 'checkpoints/easynet_rgbd_pcb2.pth')

if __name__ == '__main__':
    train()
