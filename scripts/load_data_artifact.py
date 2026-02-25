import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_artifact(path, transform, validation_split=0.2):  # real 0
    real_list = [
        "afgq",
        "celebahq",
        "coco",
        "ffhq",
        "imagemet",
        "landscape",
        "lsun",
        "metfaces",
    ]
    special = ["cyclegan"]
    train_datasets = []
    val_datasets = []
    for file in os.listdir(path):
        df = pd.read_csv(path + "/" + file + "/metadata.csv")
        image_paths = [
            path + "/" + file + "/" + imgpath for imgpath in df["image_path"]
        ]
        if file in real_list:
            labels = [1] * len(image_paths)
        elif file in special:
            labels = [0 if tg == 0 else 1 for tg in df["target"]]
        else:
            labels = [0] * len(image_paths)
        this_dataset = MyDataset(
            image_paths=image_paths, labels=labels, transform=transform
        )

        train_size = int((1 - validation_split) * len(this_dataset))
        val_size = len(this_dataset) - train_size
        this_train_dataset, this_val_dataset = random_split(
            this_dataset, [train_size, val_size]
        )
        train_datasets.append(this_train_dataset)
        val_datasets.append(this_val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print("load Artifact successfully")
    return train_dataset, val_dataset


def get_annotation_artifact(
    path, save_path1, save_path2, validation_split=0.2
):  # lasted real 1
    real_list = [
        "afgq",
        "celebahq",
        "coco",
        "ffhq",
        "imagemet",
        "landscape",
        "lsun",
        "metfaces",
    ]
    special = ["cyclegan"]
    with open(save_path1, "w") as file1, open(save_path2, "w") as file2:

        for file in os.listdir(path):
            df = pd.read_csv(path + "/" + file + "/metadata.csv")
            image_paths = [
                path + "/" + file + "/" + imgpath for imgpath in df["image_path"]
            ]
            if file in real_list:
                labels = [0] * len(image_paths)
            elif file in special:
                labels = [1 if tg == 0 else 0 for tg in df["target"]]
            else:
                labels = [1] * len(image_paths)

            train_size = int((1 - validation_split) * len(image_paths))
            val_size = len(image_paths) - train_size
            indices = np.arange(len(image_paths))
            np.random.shuffle(indices)

            # 根据随机索引划分训练集和测试集
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            # 根据索引获取图像数据和标签
            train_images = image_paths[train_indices]
            train_labels = labels[train_indices]
            test_images = image_paths[test_indices]
            test_labels = labels[test_indices]

            for i in range(len(train_images)):
                file1.write(train_images[i] + " " + str(train_labels[i] + "\n"))
            for i in range(len(test_images)):
                file2.write(test_images[i] + " " + str(test_labels[i] + "\n"))
