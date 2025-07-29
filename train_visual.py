import torch, os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import RGBDImageDataset
from autoencoder import ConvAutoencoderRGBD
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# paths full
rgb_normal = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Normal/rgb'
depth_normal = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Normal/depth'
rgb_anom = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Anomaly/rgb'
depth_anom = '/home/multimediateam/dhiep/image/pcb2/Data/Images/Anomaly/depth'

dataset = RGBDImageDataset(rgb_normal, depth_normal, rgb_anom, depth_anom)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvAutoencoderRGBD().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def visualize(inp, recon, label, epoch, idx):
    inp = inp.cpu().numpy().transpose(1,2,0)
    recon = recon.cpu().detach().numpy().transpose(1,2,0)
    err = abs(recon[:,:,:3] - inp[:,:,:3]).mean(axis=2)
    over = inp[:,:,:3].copy()
    over[:,:,0] = over[:,:,0] + err
    fig, axs = plt.subplots(1,4, figsize=(12,3))
    axs[0].imshow(inp[:,:,:3]); axs[0].set_title('Input RGB')
    axs[1].imshow(recon[:,:,:3]); axs[1].set_title('Reconstructed')
    axs[2].imshow(err, cmap='hot'); axs[2].set_title('Anomaly Map')
    axs[3].imshow(over); axs[3].set_title(f'Overlay L={label}')
    for ax in axs: ax.axis('off')
    plt.suptitle(f'Epoch {epoch} Sample {idx}')
    plt.show()

for epoch in range(1,31):
    model.train()
    total = 0
    for i,(inp, labels) in enumerate(loader):
        inp = inp.to(device)
        out = model(inp)
        loss = F.mse_loss(out[:,:3], inp[:,:3])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f'Epoch {epoch}, Loss {total/len(loader):.4f}')

    # hiển thị một mẫu anomaly nếu có
    model.eval()
    with torch.no_grad():
        for j,(inp, label) in enumerate(loader):
            if label[0]==1 or j==0: 
                visualize(inp[0], model(inp.to(device))[0], label[0].item(), epoch, j)
                break

torch.save(model.state_dict(), 'conv_rgbd_pcb2.pth')
