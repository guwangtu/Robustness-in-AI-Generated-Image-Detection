import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from load_data_artifact import load_artifact
from tqdm import tqdm

to_numpy = transforms.ToPILImage()

# 遍历数据加载器，将图像数据收集到列表中

train_transform = transforms.Compose(
    [
        transforms.RandomRotation(20),  # 随机旋转角度
        transforms.ColorJitter(brightness=0.1),  # 颜色亮度
        transforms.Resize([224, 224]),  # 设置成224×224大小的张量
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225]),
    ]
)
batch_size = 128

train_data, val_data = load_artifact(
    path="/data/user/shx/datasets/Artifact", transform=train_transform
)
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8)
image_list = []
for images, _ in tqdm(train_loader):
    for image in images:
        image_np = np.array(to_numpy(image))
        image_list.append(image_np)

# 将图像数据转换为 NumPy 数组
image_array = np.array(image_list)

# 计算均值和标准差
mean = np.mean(image_array, axis=(0, 1, 2))
std = np.std(image_array, axis=(0, 1, 2))

# 打印计算得到的均值和标准差
print("Mean:", mean)
print("Std:", std)

image_list = []
for images, _ in tqdm(val_loader):
    for image in images:
        image_np = np.array(to_numpy(image))
        image_list.append(image_np)

# 将图像数据转换为 NumPy 数组
image_array = np.array(image_list)

# 计算均值和标准差
mean = np.mean(image_array, axis=(0, 1, 2))
std = np.std(image_array, axis=(0, 1, 2))

# 打印计算得到的均值和标准差
print("Mean:", mean)
print("Std:", std)
