import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

class ImageDataset_Pixellated(Dataset):
    def __init__(self, root, transforms_=None, random_transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.random_transform = transforms.Compose(random_transforms_)
        self.files = sorted(glob.glob(os.path.join(root % mode, "*.jpg")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        # Convert grayscale images to rgb
        if img.mode != "RGB":
            img = to_rgb(img)
        w, h = img.size
        img = self.random_transform(img)
        img_A = img.copy()
        img_B = img.copy()
        img_M = img.copy()

        # Paste random patch
        patch_size_w = random.randrange(16, w)
        patch_size_h = random.randrange(16, h)
        x1 = random.randint(1, w-patch_size_w)
        y1 = random.randint(1, h-patch_size_h)
        x2 = x1+patch_size_w
        y2 = y1+patch_size_h
        # print(x1,y1,x2,y2)
        img_M = img_M.crop((x1, y1, x2, y2))
        # Create pixellated image
        pixel_size = random.randint(2, 64)
        img_M = img_M.resize(
            (
                patch_size_w//pixel_size + 1,
                patch_size_h//pixel_size + 1,
            ),
            resample=Image.BILINEAR
        )
        img_M = img_M.resize((patch_size_w, patch_size_h), Image.NEAREST)
        img_B.paste(img_M, (x1, y1))

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
