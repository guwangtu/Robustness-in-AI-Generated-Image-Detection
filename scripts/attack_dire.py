"""
Modified from compute_dire/my_compute_dire.py
"""

import argparse
import os
import torch
  

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist


from compute_dire.guided_diffusion import dist_util, logger
from compute_dire.guided_diffusion.image_datasets import load_data_for_reverse
from compute_dire.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.configure(dir=args.recons_dir)

    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)
    logger.log(str(args))

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    logger.log("have created model and diffusion")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )
    logger.log("have created data loader")
    logger.log("computing recons & DIRE ...")
    data_len=next(data) 


def compute_dire(imgs, model, diffusion,cfg):

    # imgs already on the correct device from caller
    model_kwargs = {}
    reverse_fn = diffusion.ddim_reverse_sample_loop
    
    image_size = cfg['image_size']
    batch_size = cfg['batch_size']

    imgs = reshape_image(imgs, image_size)

    latent = reverse_fn(
        model,
        (batch_size, 3, image_size, image_size),
        noise=imgs,
        clip_denoised=cfg['clip_denoised'],
        model_kwargs=model_kwargs,
        real_step=cfg['real_step'],
    )
    sample_fn = diffusion.p_sample_loop if not cfg['use_ddim'] else diffusion.ddim_sample_loop
    recons = sample_fn(
        model,
        (batch_size, 3, image_size, image_size),
        noise=latent,
        clip_denoised=cfg['clip_denoised'],
        model_kwargs=model_kwargs,
        real_step=cfg['real_step'],
    )

    dire = th.abs(imgs - recons)/2
    return dire,recons


 


def create_argparser():
    defaults = dict(
        images_dir="/data2/wangzd/dataset/DiffusionForensics/images",
        recons_dir="/data2/wangzd/dataset/DiffusionForensics/recons",
        dire_dir="/data2/wangzd/dataset/DiffusionForensics/dire",
        clip_denoised=True,
        num_samples=-1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
 