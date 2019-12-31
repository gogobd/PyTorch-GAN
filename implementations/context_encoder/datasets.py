import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(
        self,
        root,
        transforms_=None,
        img_size=128,
        mask_size=64,
        mode="train",
        kind="mask"
    ):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.kind = kind
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def get_mosaic(self, img, img_orig):
        mosaic_image = img_orig
        mosaic_size = random.randint(8, 32)
        mosaic_transform = transforms.Compose(
            [
#                transforms.Normalize((1, 1, 1), (0.5, 0.5, 0.5)),
#                transforms.ToPILImage(),
                transforms.Resize((self.img_size // mosaic_size, self.img_size // mosaic_size), Image.BICUBIC),
                transforms.Resize((self.img_size, self.img_size), Image.NEAREST),
                self.transform,
            ]
        )
        mosaic_image = mosaic_transform(mosaic_image)
        return mosaic_image

    def apply_random_mosaic(self, img, img_orig):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        mosaic_img = self.get_mosaic(img, img_orig)
        masked_img[:, y1:y2, x1:x2] = mosaic_img[:, y1:y2, x1:x2]

        return masked_img, masked_part

    def apply_center_mosaic(self, img, img_orig):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        mosaic_img = self.get_mosaic(img, img_orig)
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = mosaic_img[:, i : i + self.mask_size, i : i + self.mask_size]

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img_orig = img.copy()
        img = self.transform(img)

        if self.mode == "train":
            if self.kind == "mask":
                # For training data perform random mask
                masked_img, aux = self.apply_random_mask(img)
            elif self.kind == "mosaic":
                # For training data perform random mask
                masked_img, aux = self.apply_random_mosaic(img,img_orig)
            else:
                raise("kind not implemented.")
        else:
            if self.kind == "mask":
                # For test data mask the center of the image
                masked_img, aux = self.apply_center_mask(img)
            elif self.kind == "mosaic":
                # For test data mask the center of the image
                masked_img, aux = self.apply_center_mosaic(img, img_orig)
            else:
                raise("kind not implemented.")

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
