import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class InfraredDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0

        h, w = img.shape
        ps = self.patch_size

        if h < ps or w < ps:
            img = np.pad(
                img,
                ((0, max(0, ps - h)), (0, max(0, ps - w))),
                mode='reflect'
            )
            h, w = img.shape

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        patch = img[top:top + ps, left:left + ps]

        return torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)


if __name__ == '__main__':
    dataset = InfraredDataset(root_dir='./datasets/enhanced_images/', patch_size=128)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    print('Batch shape:', batch.shape)  # (4, 1, 128, 128)