from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import torch

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class WarpDataset(Dataset):
    def __init__(
        self,
        meta_folder,
        dataset,
        preprocess,
        split="train",
        fewshot=0,
        seed=1,
        mask_path=None,
    ):
        super(WarpDataset, self).__init__()
        if split == "test":
            index_file = os.path.join(meta_folder, dataset, "test.txt")
        elif split == "train":
            index_file = os.path.join(
                meta_folder, dataset, f"fewshot{fewshot}_seed{seed}.txt"
            )
        self.labels = []
        self.img_files = []
        self.preprocess = preprocess
        self.mask_path = mask_path
        self.vis_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    size=224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ]
        )

        with open(index_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split(" ")
                self.img_files.append(img_path)
                self.labels.append(int(label))

        labels = np.array(self.labels)
        num_classes = labels.max() + 1
        # in case some datasets have samples less than fewshot
        if split == "train":
            for lbl in range(num_classes):
                num_samples = np.sum(labels == lbl)
                if num_samples < fewshot:
                    indexes = np.where(labels == lbl)[0]
                    self.labels.extend([lbl] * (fewshot - num_samples))
                    self.img_files.extend(
                        [self.img_files[i] for i in indexes[: (fewshot - num_samples)]]
                    )
        self.labels = np.array(self.labels)
        self.indexes = np.arange(len(self.img_files))

    def get_labels(self):
        return self.labels

    def get_vis_img(self, idx):
        real_idx = self.indexes[idx]
        img_path = self.img_files[real_idx]
        vis_img = self.resize_preprocess(Image.open(img_path).convert("RGB"))
        return vis_img

    def get_img(self, idx):
        real_idx = self.indexes[idx]
        img = self.preprocess(Image.open(self.img_files[real_idx]).convert("RGB"))
        return img

    def get_mask(self, idx):
        real_idx = self.indexes[idx]
        mask_path = self.mask_path[real_idx]
        return torch.load(mask_path).float()

    def __getitem__(self, idx):
        real_idx = self.indexes[idx]
        img = self.get_img(idx)
        label = self.labels[real_idx]
        mask = torch.zeros_like(img)
        if self.mask_path is not None:
            mask = self.get_mask(idx)
        return img, label, mask

    def __len__(self):
        return len(self.indexes)
