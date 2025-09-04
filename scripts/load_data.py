import os
import shutil
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import itertools

image_end = (".jpg", ".png", ".JPEG")
real_list = ["laion", "celeba", "lfw", "selfie"]
fake_list = ["sfhq", "sdface"]
type_map = {
    "laion": 0,
    "celeba": 0,
    "lfw": 0,
    "selfie": 0,
    "sfhq": 1,
    "sdface": 1,
    "artifact": 2,
    "df": 2,
    "genimage": 1,
    "imagenet": 0,
    "fold": 2,
    "n": 2,
}  # 0真1假2混合


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
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as e:
            print(f"Error loading image: {e}")
        if self.transform:
            image = self.transform(image)

        return image, label

    def copy_img(self, save_path, real_label=0):
        if real_label == 0:
            dic = {0: "real", 1: "fake"}
        else:
            dic = {1: "real", 0: "fake"}
        os.makedirs(save_path + "/real", exist_ok=True)
        os.makedirs(save_path + "/fake", exist_ok=True)
        for i in tqdm(range(len(self.image_paths))):
            label = dic[self.labels[i]]
            img_path = self.image_paths[i]
            new_name = label + "_" + str(i) + "." + img_path.split(".")[-1]
            new_file_path = save_path + "/" + label + "/" + new_name
            shutil.copy(img_path, new_file_path)

    def shrink_dataset(self, ratio):
        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be between 0 and 1.")

        new_size = int(len(self.image_paths) * ratio)
        indices = random.sample(range(len(self.image_paths)), new_size)

        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]


def load_image_fold(path, label, transform, validation_split=0.2):
    imgs = []
    # print(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(image_end):
                imgs.append(os.path.join(root, file))
    # print('imagenum '+str(len(imgs)))
    return spilt_dataset(
        MyDataset(imgs, [label] * len(imgs), transform), validation_split
    )


def load_sfhq(path, *args, **kwargs):
    imgs = []
    for p in os.listdir(path):
        p_path = path + "/" + p
        image_path = p_path + "/" + "images"
        imgs = imgs + load_images(image_path)
    return spilt_dataset(
        MyDataset(imgs, [1] * len(imgs), kwargs["transform"]),
        kwargs["validation_split"],
    )


def load_artifact(path, *args, **kwargs):  # real 0
    artifact_real_list = [
        "afgq",
        "celebahq",
        "coco",
        "ffhq",
        "imagemet",
        "landscape",
        "lsun",
        "metfaces",
    ]
    special = ["cyclegan", "pro_gan"]
    datasets = []
    base_path = "/data_nas/users/shx/datasets/Artifact"
    """if path==base_path:
        mode=0
    else:
        mode=1"""
    for file in os.listdir(base_path):
        df = pd.read_csv(base_path + "/" + file + "/metadata.csv")
        image_paths = [
            path + "/" + file + "/" + imgpath for imgpath in df["image_path"]
        ]
        if file in artifact_real_list:
            labels = [1] * len(image_paths)
        elif file in special:
            labels = [0 if tg == 0 else 1 for tg in df["target"]]
        else:
            labels = [0] * len(image_paths)

        this_dataset = MyDataset(image_paths, labels, kwargs["transform"])
        datasets.append(this_dataset)

    print("load Artifact successfully")
    return spilt_dataset(ConcatDataset(datasets), kwargs["validation_split"])


def load_diffusion_forensics(path, *args, **kwargs):  # 0真1假
    try:
        train_transform = kwargs["train_transform"]
        val_transform = kwargs["val_transform"]
    except:
        train_transform = kwargs.get("transform", None)
        val_transform = kwargs.get("transform", None)

    train_dataset_path = path + "/train"
    train_dataset = load_diffusion_forensics_Dataset(
        train_dataset_path, train_transform
    )
    test_dataset_path = path + "/test"
    test_dataset = load_diffusion_forensics_Dataset(test_dataset_path, val_transform)
    return train_dataset, test_dataset


def load_GenImage(GenImage_path, *args, **kwargs):  # real 0

    train_datasets = []
    val_datasets = []

    genimage_list = [
        "ADM",
        "BigGAN",
        "glide",
        "Midjourney",
        "sdv4",
        "stable_diffusion_v_1_4",
        "stable_diffusion_v_1_5",
        "VQDM",
        "wukong",
    ]
    for subfolder in genimage_list:
        this_path = os.path.join(GenImage_path, subfolder)
        if not os.path.exists(this_path):
            continue
        if not os.path.exists(os.path.join(this_path, "val")):
            root, dirs, files = next(os.walk(this_path))
            this_path = os.path.join(this_path, dirs[0])
        train_path = os.path.join(this_path, "train")
        val_path = os.path.join(this_path, "val")

        train_datasets.append(load_single_dataset(train_path, kwargs["transform"]))
        val_datasets.append(load_single_dataset(val_path, kwargs["transform"]))

    print("load GenImage successfully")
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def load_fold(path, *args, **kwargs):  # 0真1假
    # 标准化数据集格式biaozhun
    # -fold
    # --train
    # ---real
    # ---fake
    train_path = path + "/train"
    test_path = path + "/test"
    if not os.path.exists(test_path):
        test_path = path + "/val"
    try:
        train_transform = kwargs["train_transform"]
        val_transform = kwargs["val_transform"]
    except:
        train_transform = kwargs.get("transform", None)
        val_transform = kwargs.get("transform", None)
    train_dataset = load_single_dataset(train_path, train_transform)
    val_dataset = load_single_dataset(test_path, val_transform)
    return train_dataset, val_dataset


load_function_map = {
    "sfhq": load_sfhq,
    "artifact": load_artifact,
    "df": load_diffusion_forensics,
    "fold": load_fold,
    "n": load_fold,
    "default": load_image_fold,
}


def get_load_function(type):
    return load_function_map.get(type, load_function_map["default"])


def spilt_dataset(dataset, validation_split=0.2):  #  共用transform
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    image_paths = [dataset.image_paths[idx] for idx in train_indices]
    labels = [dataset.labels[idx] for idx in train_indices]
    train_dataset = MyDataset(image_paths, labels, transform=dataset.transform)

    image_paths = [dataset.image_paths[idx] for idx in val_indices]
    labels = [dataset.labels[idx] for idx in train_indices]
    val_dataset = MyDataset(image_paths, labels, transform=dataset.transform)

    return train_dataset, val_dataset


def get_spilt_dataset(
    image_list, label_list, train_transform, val_transform, validation_split=0.2
):

    X_train, X_val, y_train, y_val = train_test_split(
        image_list, label_list, test_size=validation_split, random_state=42
    )

    train_dataset = MyDataset(X_train, y_train, transform=train_transform)
    val_dataset = MyDataset(X_val, y_val, transform=val_transform)

    return train_dataset, val_dataset


def balance_data(data_list, ratio_list=None):  # todo dataset 版
    if not ratio_list:
        return data_list
    ratio = [item / sum(ratio_list) for item in ratio_list]
    balanced_data = []
    total_length = sum(len(dataset) for dataset in data_list)
    target_lengths = [round(total_length * ratio) for ratio in ratios]

    for dataset, target_length in zip(data_list, target_lengths):
        if len(dataset) > target_length:
            balanced_data.append(random.sample(dataset, target_length))
        else:
            balanced_data.append(
                dataset * (target_length // len(dataset))
                + random.sample(dataset, target_length % len(dataset))
            )
    return balanced_data


def load_datasets(
    data_paths,
    data_types,
    train_transform=None,
    validation_split=0.2,
    ratio_list=None,
    concat=True,
):
    train_datasets = []
    val_datasets = []

    for i in range(len(data_paths)):
        this_path = data_paths[i]
        this_type = data_types[i]
        print("loading " + this_type)
        print("using " + get_load_function(this_type).__name__)
        train_dataset, val_dataset = get_load_function(this_type)(
            path=this_path,
            label=type_map[this_type],
            transform=train_transform,
            validation_split=validation_split,
        )
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        print(
            "dataset_size:"
            + str(len(train_dataset))
            + " + "
            + str(len(val_dataset))
            + " = "
            + str(len(train_dataset) + len(val_dataset))
        )
    # bd = balance_data(data_list,ratio_list)
    if concat:
        return ConcatDataset(train_datasets), ConcatDataset(val_datasets)
    else:
        return train_datasets, val_datasets


def load_data_2_path(
    real_path, fake_path, train_transform, val_transform, each_class_num=0
):
    real_pic = load_image_fold(real_path)
    fake_pic = load_image_fold(fake_path)

    if each_class_num == 0:
        each_class_num = (
            len(real_pic) if len(real_pic) >= len(fake_pic) else len(fake_pic)
        )
    real_pic = resize_list(real_pic, each_class_num)
    fake_pic = resize_list(fake_pic, each_class_num)

    real_data = MyDataset(real_pic, [0] * len(real_pic), train_transform)
    fake_data = MyDataset(fake_pic, [1] * len(fake_pic), train_transform)

    raw_data = ConcatDataset([real_data, fake_data])

    return spilt_dataset(raw_data)


def load_images(path):
    imgs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(image_end):
                imgs.append(os.path.join(root, file))
    return imgs


def load_single_dataset(path, transform=None):  # 0真1假
    file_real_list = ["real", "nature"]
    datasets = []
    for file in os.listdir(path):
        if "." in file:
            continue
        if file in file_real_list:
            label = [0]
        else:
            label = [1]
        fold_path = path + "/" + file
        images = load_images(fold_path)
        labels = label * len(images)

        datasets.append(MyDataset(images, labels, transform))
    return ConcatDataset(datasets)


def resize_list(input_list, crop_size):

    original_length = len(input_list)

    if original_length > crop_size:
        return list(np.random.choice(input_list, size=crop_size, replace=False))
    elif original_length == crop_size:
        return input_list
    else:
        # 计算需要复制的次数
        num_copies = crop_size // original_length
        remainder = crop_size % original_length

        # 创建新的列表
        new_list = input_list * num_copies

        # 如果还有剩余，则随机选取剩余元素
        if remainder > 0:
            new_list.extend(np.random.choice(input_list, size=remainder, replace=False))

        return new_list


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


def load_diffusion_forensics_Dataset(path, transform):
    datasets = []
    for file in os.listdir(path):
        if "." in file:
            continue
        file_path = path + "/" + file
        datasets.append(load_single_dataset(file_path, transform))
    return ConcatDataset(datasets)


def ConcatDataset(datasets):
    if len(datasets) == 0:
        return None
    combined_image_paths = []
    combined_labels = []

    transform = None
    for dataset in datasets:
        if dataset == None:
            continue
        transform = dataset.transform
        combined_image_paths.extend(dataset.image_paths)
        combined_labels.extend(dataset.labels)

    # 创建一个新的 MyDataset 实例
    combined_dataset = MyDataset(combined_image_paths, combined_labels, transform)
    return combined_dataset
