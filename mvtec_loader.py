# mvtec_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        """
        Args:
            root_dir (str): Đường dẫn đến lớp đối tượng, ví dụ: 'image/bottle'
            phase (str): 'train' hoặc 'test'
            transform (callable): torchvision.transforms để resize, normalize...
        """
        self.phase = phase
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.img_paths = []
        self.labels = []  # 0 = good, 1 = anomaly

        target_dir = os.path.join(root_dir, phase)
        categories = os.listdir(target_dir)

        for category in categories:
            category_path = os.path.join(target_dir, category)
            if not os.path.isdir(category_path):
                continue

            label = 0 if category == "good" else 1

            for img_name in os.listdir(category_path):
                if img_name.endswith((".png", ".jpg")):
                    self.img_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label