
import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def safe_collate(batch):
    # batch: List[Tuple[image, mask]]
    imgs, masks = zip(*batch)
    # 统一为普通、可resize的张量，并保证连续/可写
    imgs  = [torch.as_tensor(i).contiguous().clone() for i in imgs]
    masks = [torch.as_tensor(m).contiguous().clone() for m in masks]
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)

class PairedData(Dataset):
    def __init__(self, root, target='train', use_num=-1,
                 img_size=(256, 256),       # 统一尺寸
                 rgb=True,                  # 本方案用 RGB
                 mean=None, std=None):
        super().__init__()
        self.root = root
        self.target = target
        self.img_size = img_size
        self.rgb = rgb
        C = 3 if rgb else 1
        self.normalize = transforms.Normalize(mean or [0.5]*C, std or [0.5]*C)

        image_folder = os.path.join(root, target, "image")
        label_folder = os.path.join(root, target, "label")
        if not os.path.isdir(image_folder): raise FileNotFoundError(image_folder)
        if not os.path.isdir(label_folder): raise FileNotFoundError(label_folder)

        self.image_path, self.label_path = [], []
        names = sorted([n for n in os.listdir(image_folder)
                        if os.path.isfile(os.path.join(image_folder, n))])

        for n in names:
            img = os.path.join(image_folder, n)
            base, _ = os.path.splitext(n)

            # 匹配 *_segmentation.*
            cands = [os.path.join(label_folder, f"{base}_Segmentation{ext}")
                     for ext in [".png", ".jpg", ".jpeg"]]
            lbl = next((p for p in cands if os.path.exists(p)), None)
            if lbl is None:
                wild = glob.glob(os.path.join(label_folder, f"{base}_Segmentation.*")) or \
                       glob.glob(os.path.join(label_folder, f"{base}.*"))
                if wild: lbl = sorted(wild, key=lambda x: (os.path.splitext(x)[1].lower() != ".jpg", x))[0]
            if lbl is None: continue

            self.image_path.append(img)
            self.label_path.append(lbl)
            if use_num > 0 and len(self.image_path) >= use_num: break

        if len(self.image_path) == 0:
            raise ValueError(f"No valid pairs under {image_folder} & {label_folder}")

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        ip, lp = self.image_path[idx], self.label_path[idx]
        img = Image.open(ip).convert('RGB' if self.rgb else 'L')
        msk = Image.open(lp).convert('L')

        # 统一尺寸
        if self.img_size is not None:
            img = F.resize(img, self.img_size, InterpolationMode.BILINEAR)
            msk = F.resize(msk, self.img_size, InterpolationMode.NEAREST)

        # 张量化与归一化
        img_t = F.to_tensor(img)
        img_t = self.normalize(img_t)   # [3,H,W]

        # mask -> [H,W] long(0/1)
        m = np.array(msk, dtype=np.uint8)
        m = (m > 0).astype(np.int64)
        m_t = torch.from_numpy(m)       # [H,W] long

        return img_t, m_t