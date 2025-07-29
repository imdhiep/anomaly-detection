import torch, os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import RGBDImageDataset
from model import EasyNet
import torch.nn.functional as F

def show_anomaly_visual(inp, recon, amap, idx):
    inp = inp[:3].cpu().permute(1,2,0).numpy()
    rec = recon[:3].cpu().permute(1,2,0).detach().numpy()
    amap = amap.squeeze().cpu().detach().numpy()
    err = abs(rec - inp).mean(axis=2)
    overlay = inp.copy()
    overlay[:,:,0] = (overlay[:,:,0] + err).clip(0,1)
    fig, axs = plt.subplots(1,4,figsize=(16,4))
    axs[0].imshow(inp); axs[0].set_title('Original RGB')
    axs[1].imshow(rec); axs[1].set_title('Reconstruction')
    axs[2].imshow(amap, cmap='hot'); axs[2].set_title('Anomaly Map')
    axs[3].imshow(overlay); axs[3].set_title('Overlay')
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"anomaly_vis_{idx}.png")
    print(f"Saved anomaly_vis_{idx}.png")

def test(num_samples=3):
    rgb_n = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Normal/rgb'
    depth_n = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Normal/depth'
    rgb_a = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Anomaly/rgb'
    depth_a = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Anomaly/depth'

    ds = RGBDImageDataset(rgb_n, depth_n, rgb_a, depth_a, transform_size=(128,128))
    print("Total samples:", len(ds), "anomalies:", sum(1 for _, label in ds if label==1))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EasyNet().to(device)
    ckpt = torch.load('checkpoints/easynet_rgbd_pcb2.pth', map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    count = 0
    with torch.no_grad():
        for idx, (inp, label) in enumerate(loader):
            if label.item() != 1:
                continue
            inp = inp.to(device)
            recon, amap = model(inp)
            recon = F.interpolate(recon, size=inp.shape[2:], mode='bilinear', align_corners=False)
            show_anomaly_visual(inp[0], recon[0], amap[0], idx)
            count += 1
            if count >= num_samples:
                break

if __name__ == '__main__':
    test(3)
