from PIL import Image, ImageFilter
import os
from tqdm import tqdm
import torchvision.transforms as transforms

# 读取图片集
src_dir = "/data/user/shx/Generate_image_detection/LASTED/data/face_dataset/test"
output_dir = "/data/user/shx/Generate_image_detection/face_test_70_com"

quality = 65
blur = 5

os.makedirs(output_dir, exist_ok=True)

# # JPEG压缩
# for file in tqdm(os.listdir(src_dir)):
#     with Image.open(os.path.join(src_dir, file)) as img:
#         name, ext = os.path.splitext(file)
#         save_path = os.path.join(output_dir, name + '.jpg')
#         img.convert('RGB').save(save_path, "JPEG",  quality=quality)


# Blur
"""for subfolder in os.listdir(src_dir):
    subfolder_dir=os.path.join(src_dir,subfolder)
    os.mkdir(output_dir+'/'+subfolder, exist_ok=True)
    for file in os.listdir(subfolder_dir):
        with Image.open(os.path.join(subfolder_dir, file)) as img:
            name, ext = os.path.splitext(file)
            save_path = os.path.join(output_dir+'/'+subfolder, name + '.jpg')
            img = img.filter(ImageFilter.GaussianBlur(blur))
            img.save(save_path)"""
for subfolder in os.listdir(src_dir):
    subfolder_dir = os.path.join(src_dir, subfolder)
    os.makedirs(output_dir + "/" + subfolder, exist_ok=True)
    for file in os.listdir(subfolder_dir):
        with Image.open(os.path.join(subfolder_dir, file)) as img:
            name, ext = os.path.splitext(file)
            save_path = os.path.join(output_dir + "/" + subfolder, name + ".jpg")

            image_pil = img
            image_pil.save(save_path, optimize=True, quality=70)


output_dir = "/data/user/shx/Generate_image_detection/face_test_35_com"
os.makedirs(output_dir, exist_ok=True)
for subfolder in os.listdir(src_dir):
    subfolder_dir = os.path.join(src_dir, subfolder)
    os.makedirs(output_dir + "/" + subfolder, exist_ok=True)
    for file in os.listdir(subfolder_dir):
        with Image.open(os.path.join(subfolder_dir, file)) as img:
            name, ext = os.path.splitext(file)
            save_path = os.path.join(output_dir + "/" + subfolder, name + ".jpg")

            image_pil = img
            image_pil.save(save_path, optimize=True, quality=35)
