from compute_dire.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from argparse import Namespace

import torch
import re
import os
import numpy as np


def get_imagenet_dm_conf(
    class_cond=False,
    respace="",
    device="cuda",
    model_path="/data/user/shx/Generate_image_detection/guided-diffusion/models/256x256_diffusion_uncond.pt",
):

    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
    )

    model_config = dict(
        use_fp16=False,
        attention_resolutions="32, 16, 8",
        class_cond=class_cond,
        diffusion_steps=1000,
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_scale_shift_norm=True,
        timestep_respacing=respace,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(model_config)
    args = Namespace(**defaults)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # load ckpt

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    return model, diffusion


def label_to_str(label, real=0):
    if label == real:
        s = "real"
    else:
        s = "fake"
    return s


def get_index_path(save_path):
    int_files = [int(file) for file in os.listdir(save_path)]
    if len(int_files) == 0:
        save_path = os.path.join(save_path, "1")
    else:
        save_path = os.path.join(save_path, str(max(int_files) + 1))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def find_max_num_png(folder_path):
    # 初始化最大数字标号为 0
    max_number = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为 PNG 格式
        if filename.endswith(".png"):
            # 使用正则表达式提取数字标号
            number_match = re.search(r"(\d+)\.png", filename)
            if number_match:
                number = int(number_match.group(1))
                # 更新最大数字标号
                if number > max_number:
                    max_number = number
    return max_number


def get_crt_num(target, label):
    max_value, max_index = torch.max(target, 1)
    pred_label = max_index.cpu().numpy()
    true_label = label.cpu().numpy()
    return np.sum(pred_label == true_label)
