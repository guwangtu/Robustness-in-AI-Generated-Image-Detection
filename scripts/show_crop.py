from PIL import Image
import os

def center_crop_square(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size
        min_dim = min(width, height)

        # 计算剪切区域
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2

        # 剪切图像
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(output_path)

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            center_crop_square(input_path, output_path)
            print(f'Processed: {filename}')

input_directory = '/data_nas/users/shx/datasets_n/show/adv/celeba'  # 替换为你的输入图片目录
output_directory = '/data_nas/users/shx/datasets_n/show/adv/celeba2'  # 替换为你的输出图片目录

process_images(input_directory, output_directory)