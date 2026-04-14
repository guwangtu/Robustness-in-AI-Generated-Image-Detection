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
    if cfg.get('shrink_size'):
        shrink_ratio = cfg['shrink_size']/len(val_data)
        val_data.shrink_dataset(shrink_ratio) # 只测试400张
    t_loader = DataLoader(
                val_data,
                batch_size=cfg['batch_size']
            )
    
    allnum=0    
    rightnum=0
    rightnum_adv=0
    attack_success_num=0
    detector.eval()

    for i, (image, label) in tqdm(enumerate(t_loader), total=len(t_loader)):
        # 跳过最后一回合不足一个 batch 的情况
        if image.shape[0] != cfg['batch_size']:
            continue
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
            eps=cfg.get('atk_eps', 8/255),
            alpha=cfg.get('atk_alpha', 1/255),
            steps=cfg.get('atk_steps', 10),
        )
        dire,_ = compute_dire(image*2-1, model, diffusion, cfg)
        adv_dire,_ = compute_dire(adv*2-1, model, diffusion, cfg)
        save_attack_results(image, adv, dire, adv_dire, save_root=cfg['img_save_path']+'_ce_'+str(cfg['ce_loss_weight'])+'_dire_'+str(cfg['dire_loss_weight']))



        logits = detector(dire)
        adv_logits = detector(adv_dire)
        allnum+=cfg['batch_size']
        pred = torch.argmax(logits, dim=1)
        adv_pred = torch.argmax(adv_logits, dim=1)

        rightnum += (pred == label).sum().item()
        rightnum_adv += (adv_pred == label).sum().item()
        attack_success_num += ((pred == label) & (adv_pred != label)).sum().item()
        #print(f"image {i}: clean label={label.item()}, clean pred={torch.argmax(logits, dim=1).item()}, adv pred={torch.argmax(adv_logits, dim=1).item()}")
        
    if allnum > 0:
        print(f"accuracy: {rightnum}/{allnum}={rightnum/allnum:.4f}")
        print(f"accuracy after attack: {rightnum_adv}/{allnum}={rightnum_adv/allnum:.4f}")
        if rightnum > 0:
            print(f"attack success rate: {attack_success_num}/{rightnum}={attack_success_num/rightnum:.4f}")
        else:
            print("attack success rate: N/A (no correctly classified samples)")
    else:
        print("No complete batches were processed.")
from torchvision.utils import save_image
def save_attack_results(image, adv, dire, adv_dire, save_root):

    os.makedirs(save_root, exist_ok=True)

    # 找已有最大编号
    existing = [
        int(name) for name in os.listdir(save_root)
        if name.isdigit() and os.path.isdir(os.path.join(save_root, name))
    ]

    start_idx = max(existing) + 1 if existing else 1

    B = image.shape[0]

    for i in range(B):

        folder = os.path.join(save_root, str(start_idx + i))
        os.makedirs(folder, exist_ok=True)

        save_image(image[i].detach().cpu(), os.path.join(folder, "image.png"))
        save_image(adv[i].detach().cpu(), os.path.join(folder, "adv.png"))
        save_image(dire[i].detach().cpu(), os.path.join(folder, "dire.png"))
        save_image(adv_dire[i].detach().cpu(), os.path.join(folder, "adv_dire.png"))
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
        with torch.no_grad():
            dire, recons = compute_dire(image, model, diffusion, cfg)

        ctx.save_for_backward(image, recons)
        return dire

    @staticmethod
    def backward(ctx, grad_output):
        image, recons = ctx.saved_tensors

        with torch.enable_grad():
            image_ = image.detach().requires_grad_(True)
            recons_const = recons.detach()

            eps_smooth = 1e-6
            dire_tilde = torch.sqrt((image_ - recons_const) ** 2 + eps_smooth)

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
    steps=10,
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

    eot_steps = cfg.get("eot_steps", 1)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        # diffusion 输入通常是 [-1,1]
        x_in = x * 2 - 1


        loss_total=0
        for _ in range(eot_steps):
            dire = compute_dire_bpda(x_in, model, diffusion, cfg)
            logits = detector(dire)

            if targeted:
                assert target_label is not None
                ce_loss = F.cross_entropy(logits, target_label)
                dire_scalar = dire.mean(dim=(1,2,3))
                sign = 1 - 2 * true_label.float()
                dire_loss = (dire_scalar * sign).mean()
                a=cfg['ce_loss_weight']
                b=cfg['dire_loss_weight']
                loss_total = loss_total + a*ce_loss + b*dire_loss 

            else:
                loss_total = loss_total + F.cross_entropy(logits, true_label)

        loss = loss_total / eot_steps
        grad = torch.autograd.grad(loss, x)[0]

        if targeted:
            x = x - alpha * grad.sign()
        else:
            x = x + alpha * grad.sign()

        # project to linf ball
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
        x = x.clamp(0, 1)

    return x.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    args, unknown = parser.parse_known_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg["todo"] == "attack":
        print("Running attack_dire...")
        attack_dire(cfg,device)


if __name__ == "__main__":
    main()