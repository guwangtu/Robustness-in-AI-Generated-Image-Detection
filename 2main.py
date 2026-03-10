import argparse
import os
import subprocess
import yaml
import sys
from tqdm import tqdm
from scripts.attack_dire import compute_dire

from PIL import Image
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset


from compute_dire.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from compute_dire.guided_diffusion import dist_util, logger


def attack_dire(cfg,device):
    defaults=model_and_diffusion_defaults()
    defaults.update(cfg)
    model, diffusion = create_model_and_diffusion(**args_to_dict(defaults, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(defaults['model_path'], map_location="cpu"))
    if defaults['use_fp16']:
        model.convert_to_fp16()
    model.to(device)


    detector = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    detector.fc = torch.nn.Linear(2048, 2)
    if cfg['load_path'] is not None:
        load_path =cfg['load_path']
        m_state_dict = torch.load(load_path, map_location="cpu")
        detector.load_state_dict(m_state_dict)
    detector = detector.to(device, non_blocking=True).eval()
    if cfg.get('attack_one'):
        image = Image.open("test\\91.png").convert("RGB")
        image = transforms.ToTensor()(image).unsqueeze(0)
        image= transforms.Resize((256,256))(image)
        image = image.clone().detach().to(device).requires_grad_(True)
        true_y = torch.tensor([1], device=device)    # 假设原图是 fake
        target_y = torch.tensor([0], device=device)  # 目标是 real

        adv01 = pgd_bpda_attack(
            image=image,
            true_label=true_y,
            target_label=target_y,
            targeted=True,
            model=model,
            diffusion=diffusion,
            cfg=cfg,
            detector=detector,
            eps=8/255,
            alpha=1/255,
            steps=40,
        )

        # 对比攻击前后预测
        with torch.no_grad():
            logits_clean = detector(compute_dire_bpda(image*2-1, model, diffusion, cfg))
            logits_adv   = detector(compute_dire_bpda(adv01*2-1, model, diffusion, cfg))
        print("clean logits:", logits_clean)
        print("adv logits  :", logits_adv)
        print("clean probs:", torch.softmax(logits_clean, dim=1))
        print("adv probs  :", torch.softmax(logits_adv, dim=1))

        show_image_and_adv(image, adv01, titles=("Original", "Adversarial"))  
    else:
        from scripts.load_data import load_fold
        train_transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ]
        )
        train_data, val_data = load_fold(
            cfg['dataset_path'],
            train_transform=train_transform,
            val_transform=train_transform
        )
        t_loader = DataLoader(
                    val_data,
                    batch_size=cfg['batch_size']
                )
        
        allnum=0    
        rightnum=0
        attack_success_num=0
        detector.eval()

        for i, (image, label) in tqdm(enumerate(t_loader), total=len(t_loader)):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            adv = pgd_bpda_attack(
                image=image,
                true_label=label,
                target_label=1-label,
                targeted=True,
                model=model,
                diffusion=diffusion,
                cfg=cfg,
                detector=detector,
                eps=8/255,
                alpha=1/255,
                steps=40,
            )
            dire = compute_dire(image, model, diffusion, cfg)
            adv_dire = compute_dire(adv, model, diffusion, cfg)
            logits = detector(dire)
            adv_logits = detector(adv_dire)
            allnum+=cfg['batch_size']
            pred = torch.argmax(logits, dim=1)
            adv_pred = torch.argmax(adv_logits, dim=1)

            rightnum += (pred == label).sum().item()

            attack_success_num += ((pred == label) & (adv_pred != label)).sum().item()
            #print(f"image {i}: clean label={label.item()}, clean pred={torch.argmax(logits, dim=1).item()}, adv pred={torch.argmax(adv_logits, dim=1).item()}")
            
        print(f"accuracy: {rightnum}/{allnum}={rightnum/allnum:.4f}")
        print(f"attack success rate: {attack_success_num}/{allnum}={attack_success_num/allnum:.4f}")


def test_dire(cfg):

    defaults=model_and_diffusion_defaults()
    defaults.update(cfg)
    model, diffusion = create_model_and_diffusion(**args_to_dict(defaults, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(defaults['model_path'], map_location="cpu"))
    if defaults['use_fp16']:
        model.convert_to_fp16()
    model.to("cuda"if torch.cuda.is_available() else "cpu")
 

    detector = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    detector.fc = torch.nn.Linear(2048, 2)
    if cfg['load_path'] is not None:
        load_path =cfg['load_path']
        m_state_dict = torch.load(load_path, map_location="cpu")
        detector.load_state_dict(m_state_dict)
    detector = detector.to("cuda", non_blocking=True)

    print(detector(dire).shape,dire.shape,detector(dire)) # y: shape[1,2]

    train_transform = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )
    from scripts.load_data import load_fold
    train_data, val_data = load_fold(
        "E:\\datasets\\face\\celeba",
        train_transform=train_transform,
        val_transform=train_transform
    )
    val_data.shrink_dataset(0.01) # 只测试400张
    t_loader = DataLoader(
                val_data,
                batch_size=1
            )
    
    allnum=0    
    rightnum=0
    detector.eval()

    for i, (image, label) in tqdm(enumerate(t_loader), total=len(t_loader)):
        image = image.to("cuda", non_blocking=True)
        label = label.to("cuda", non_blocking=True)
        dire = compute_dire(image, model, diffusion, cfg)
        logits = detector(dire)
        allnum+=1
        if torch.argmax(logits, dim=1) == label:
            rightnum+=1
    print(f"accuracy: {rightnum}/{allnum}={rightnum/allnum:.4f}")

import matplotlib.pyplot as plt

def show_image_and_adv(image, adv, titles=("Original", "Adversarial")):
    # [1,3,H,W] -> [H,W,3]
    img = image.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    adv_img = adv.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(titles[0])
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(adv_img)
    plt.title(titles[1])
    plt.axis("off")

    plt.tight_layout()
    plt.show()

class BPDA_DIRE(Function):
    @staticmethod
    def forward(ctx, image, model, diffusion, cfg):
        # 保存输入（用于 backward）
        ctx.save_for_backward(image)   # ✅ 正确用法：调用方法
        ctx.model = model
        ctx.diffusion = diffusion
        ctx.cfg = cfg

        # forward 用真实 compute_dire（内部 no_grad 也没关系）
        with torch.no_grad():
            dire,_ = compute_dire(image, model, diffusion, cfg)
        return dire

    @staticmethod
    def backward(ctx, grad_output):
        # ✅ 正确取回保存的张量
        (image,) = ctx.saved_tensors

        # surrogate：straight-through（identity）
        image_ = image.detach().requires_grad_(True)
        dire_tilde = image_  # STE

        grad_image = torch.autograd.grad(
            outputs=dire_tilde,
            inputs=image_,
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]

        return grad_image, None, None, None

class BPDA_DIRE_v2(Function):
    @staticmethod
    def forward(ctx, image, model, diffusion, cfg):
        # 真实 forward：算真实 dire 和真实 recons
        with torch.no_grad():
            dire, recons = compute_dire(image, model, diffusion, cfg)

        # backward 需要 image 和 recons
        ctx.save_for_backward(image, recons)
        return dire

    @staticmethod
    def backward(ctx, grad_output):
        image, recons = ctx.saved_tensors

        # 构造新的可导变量
        image_ = image.detach().requires_grad_(True)

        # stopgrad(recons)：把 recons 视为常量
        recons_const = recons.detach()

        # surrogate: |image - stopgrad(recons)|
        dire_tilde = torch.abs(image_ - recons_const)

        grad_image = torch.autograd.grad(
            outputs=dire_tilde,
            inputs=image_,
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]

        return grad_image, None, None, None


def compute_dire_bpda(image, model, diffusion, cfg):
    v2=cfg.get("attack_version", 1) == 2
    if v2:
        return BPDA_DIRE_v2.apply(image, model, diffusion, cfg)
    else:
        return BPDA_DIRE.apply(image, model, diffusion, cfg)

def pgd_bpda_attack(
    image,                # 输入：[B,3,H,W] in [0,1]
    true_label,             # 真实标签 tensor([..])，用于 untargeted
    model, diffusion, cfg,
    detector,
    eps=8/255,
    alpha=1/255,
    steps=40,
    targeted=False,
    target_label=None,      # targeted=True 时必填
    random_start=True,
):
    detector.eval()
    model.eval()

    x0 = image.detach()

    # 随机初始化（推荐）
    if random_start:
        delta = torch.empty_like(x0).uniform_(-eps, eps)
        x = (x0 + delta).clamp(0, 1)
    else:
        x = x0.clone()

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        # diffusion 输入通常是 [-1,1]
        x_in = x * 2 - 1

        dire = compute_dire_bpda(x_in, model, diffusion, cfg)  # 你已有
        logits = detector(dire)

        if targeted:
            assert target_label is not None
            loss = torch.nn.CrossEntropyLoss()(logits, target_label)
            # targeted: 最小化目标标签 loss
            grad = torch.autograd.grad(loss, x)[0]
            x = x - alpha * grad.sign()
        else:
            loss = torch.nn.CrossEntropyLoss()(logits, true_label)
            # untargeted: 最大化真实标签 loss
            grad = torch.autograd.grad(loss, x)[0]
            x = x + alpha * grad.sign()

        # 投影到 eps-ball
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
        x = x.clamp(0, 1)

    return x.detach()


def test(cfg):
     

    detector = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    detector.fc = torch.nn.Linear(2048, 2)
    if cfg['load_path'] is not None:
        load_path =cfg['load_path']
        m_state_dict = torch.load(load_path, map_location="cpu")
        detector.load_state_dict(m_state_dict)
    detector = detector.to("cuda", non_blocking=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )
    from scripts.load_data import load_image_fold
    train_data, val_data = load_image_fold(
        "E:\\datasets\\face\\celeba",
        0,
        train_transform,
        0.002
    )
    t_loader = DataLoader(
                val_data,
                batch_size=1
            )
    
    allnum=0    
    rightnum=0
    detector.eval()

    for i, (image, label) in tqdm(enumerate(t_loader), total=len(t_loader)):
        image = image.to("cuda", non_blocking=True)
        label = label.to("cuda", non_blocking=True) 
        logits = detector(image)
        allnum+=1
        if torch.argmax(logits, dim=1) == label:
            rightnum+=1
    print(f"accuracy: {rightnum}/{allnum}={rightnum/allnum:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    args, unknown = parser.parse_known_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg["todo"] == "attack":
        attack_dire(cfg,device)
    elif cfg["todo"] == "test_dire":
        test_dire(cfg)
    elif cfg["todo"] == "test":
        test(cfg)

if __name__ == "__main__":
    main()