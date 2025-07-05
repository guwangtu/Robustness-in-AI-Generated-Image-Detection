import os
from PIL import Image

# 源文件夹路径
# src_dir = '/data_nas/users/shx/datasets/DiffusionForensics/images'
src_dir = "/data/user/shx/Generate_image_detection/2image_dataset"

# 目标文件夹路径
# dst_dir = '/data_nas/users/shx/datasets/DiffusionForensics_JPEG'
dst_dir = "/data/user/shx/Generate_image_detection/imagenet_JPEG"


def convert_to_jpeg(src_path, dst_path):
    """
    将 PNG 或 JPEG 图像转换为 JPEG 格式并保存
    """
    try:
        img = Image.open(src_path)
        img.save(dst_path, "JPEG", quality=95)
        print(f"Converted {os.path.basename(src_path)} to {os.path.basename(dst_path)}")
    except Exception as e:
        print(f"Error converting {os.path.basename(src_path)}: {e}")


def copy_directory_structure(src_dir, dst_dir):
    """
    递归创建与源文件夹结构相同的目标文件夹
    """
    # root, dirs, files = next(os.walk(src_dir))
    for root, dirs, files in os.walk(src_dir):
        for dir in dirs:
            new_dir = os.path.join(
                dst_dir, os.path.relpath(os.path.join(root, dir), src_dir)
            )
            os.makedirs(new_dir, exist_ok=True)
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, os.path.relpath(src_path, src_dir))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".JPEG")):
                new_file = os.path.splitext(file)[0] + ".JPEG"
                dst_path = os.path.join(os.path.dirname(dst_path), new_file)
                convert_to_jpeg(src_path, dst_path)


# 创建目标文件夹结构
copy_directory_structure(src_dir, dst_dir)
