import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        """
        Args:
            root_dir (str): Đường dẫn đến thư mục object class, ví dụ: 'image/carrot'
            phase (str): 'train' hoặc 'test'
            transform (callable, optional): torchvision.transforms để resize, normalize,...
        """
        self.phase = phase
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.img_paths = []
        self.labels = []  # 0 = good, 1 = anomaly

        # === Xử lý train ===
        if phase == "train":
            rgb_dir = os.path.join(root_dir, "train", "good", "rgb")
            if not os.path.exists(rgb_dir):
                raise ValueError(f"Không tìm thấy thư mục: {rgb_dir}")
            image_files = glob(os.path.join(rgb_dir, "*.png"))
            self.img_paths = image_files
            self.labels = [0] * len(image_files)

        # === Xử lý test ===
        elif phase == "test":
            test_root = os.path.join(root_dir, "test")
            if not os.path.exists(test_root):
                raise ValueError(f"Không tìm thấy thư mục: {test_root}")
            defect_types = os.listdir(test_root)

            for defect_type in defect_types:
                rgb_dir = os.path.join(test_root, defect_type, "rgb")
                if not os.path.exists(rgb_dir):
                    continue
                image_files = glob(os.path.join(rgb_dir, "*.png"))
                label = 0 if defect_type == "good" else 1
                self.img_paths.extend(image_files)
                self.labels.extend([label] * len(image_files))
        else:
            raise ValueError("phase phải là 'train' hoặc 'test'.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
