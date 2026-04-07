"""
将 df_imagenet 数据集中所有图片统一转为 JPEG (quality=95)，
保持目录结构不变，保存到新路径。
"""
import os
import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

SRC = r"E:\datasets\df_imagenet"
DST = r"E:\datasets\df_imagenet_p"
QUALITY = 95


def convert_one(args):
    src_path, dst_path = args
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img = Image.open(src_path).convert("RGB")
        img.save(dst_path, format="JPEG", quality=QUALITY)
        return True
    except Exception as e:
        return f"FAIL {src_path}: {e}"


def collect_files(src_root, dst_root):
    tasks = []
    for root, dirs, files in os.walk(src_root):
        for f in files:
            src_path = os.path.join(root, f)
            rel = os.path.relpath(src_path, src_root)
            # 统一输出为 .jpg 后缀
            dst_path = os.path.join(dst_root, os.path.splitext(rel)[0] + ".jpg")
            tasks.append((src_path, dst_path))
    return tasks


def main():
    tasks = collect_files(SRC, DST)
    print(f"Total files to convert: {len(tasks)}")

    workers = min(8, os.cpu_count() or 4)
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(convert_one, tasks, chunksize=64), total=len(tasks)):
            if result is not True:
                failed.append(result)

    if failed:
        print(f"\n{len(failed)} failures:")
        for f in failed[:20]:
            print(f)
    else:
        print("All done, no failures.")


if __name__ == "__main__":
    main()
