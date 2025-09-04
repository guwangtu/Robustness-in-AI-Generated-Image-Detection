"""from load_data import load_image_fold

path = "/data_nas/users/shx/datasets/DiffusionForensics/image_test/test/image"
image = load_image_fold(path)
print(len(image))"""

import torch

a = torch.tensor([1, 2, 3, 4, 5])
# a.to(device="cuda:7", non_blocking=True)

a.to(device="cuda:7")
