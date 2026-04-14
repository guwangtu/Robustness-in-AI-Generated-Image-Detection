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


def adv_train_dire(cfg, device):
    # ===== 加载 diffusion 模型 =====
    defaults = model_and_diffusion_defaults()
    defaults.update(cfg)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(defaults, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(defaults['model_path'], map_location="cpu")
    )
    if defaults['use_fp16']:
        model.convert_to_fp16()
    model.to(device)
    model.eval()

    # ===== 加载 detector =====
    detector = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    detector.fc = torch.nn.Linear(2048, 2)
    if cfg.get('load_path') is not None:
        m_state_dict = torch.load(cfg['load_path'], map_location="cpu")
        detector.load_state_dict(m_state_dict)
    detector = detector.to(device, non_blocking=True)

    # ===== 数据集 =====
    from scripts.load_data import load_fold, ConcatDataset
    train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # 支持 dataset_paths (列表) 或 dataset_path (单个)，多个数据集自动合并
    dataset_paths = cfg.get('dataset_paths', None)
    if dataset_paths is None:
        dataset_paths = [cfg['dataset_path']]
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    train_datasets = []
    val_datasets = []
    for p in dataset_paths:
        t, v = load_fold(p, train_transform=train_transform, val_transform=train_transform)
        train_datasets.append(t)
        val_datasets.append(v)
        print(f"  loaded {p}: train={len(t)}, val={len(v)}")

    train_data = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_data = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    print(f"Total: train={len(train_data)}, val={len(val_data)}")

    # epoch_sample_size: 每个 epoch 随机采样的图片数量，None 或 0 表示使用全量
    epoch_sample_size = cfg.get('epoch_sample_size', None)

    val_loader = DataLoader(val_data, batch_size=cfg['batch_size'])

    # ===== 训练超参 =====
    epochs = cfg.get('epochs', 10)
    lr = cfg.get('lr', 1e-4)
    adv_ratio = cfg.get('adv_ratio', 0.5)  # 对抗样本 loss 权重
    save_path = cfg.get('save_path', 'checkpoint/adv_train')
    os.makedirs(save_path, exist_ok=True)

    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ========== 每个 epoch 随机采样子集 ==========
        if epoch_sample_size and epoch_sample_size < len(train_data):
            indices = torch.randperm(len(train_data))[:epoch_sample_size].tolist()
            epoch_subset = Subset(train_data, indices)
            print(f"Epoch {epoch+1}: randomly sampled {len(epoch_subset)}/{len(train_data)} images")
        else:
            epoch_subset = train_data
        train_loader = DataLoader(epoch_subset, batch_size=cfg['batch_size'], shuffle=True)

        # ========== Train ==========
        detector.train()
        total_loss = 0.0
        clean_correct = 0
        adv_correct = 0
        total_samples = 0

        for i, (image, label) in tqdm(enumerate(train_loader), total=len(train_loader),
                                       desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if image.shape[0] != cfg['batch_size']:
                continue
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 生成 BPDA 对抗样本（detector 设为 eval 内部会自动处理）
            adv = pgd_bpda_attack(
                image=image,
                true_label=label,
                target_label=1 - label,
                targeted=True,
                model=model,
                diffusion=diffusion,
                cfg=cfg,
                detector=detector,
                eps=cfg.get('atk_eps', 8/255),
                alpha=cfg.get('atk_alpha', 1/255),
                steps=cfg.get('atk_steps', 10),
            )

            detector.train()

            # 计算 clean DIRE 和 adv DIRE
            with torch.no_grad():
                dire, _ = compute_dire(image * 2 - 1, model, diffusion, cfg)
                adv_dire, _ = compute_dire(adv * 2 - 1, model, diffusion, cfg)

            # clean loss
            clean_logits = detector(dire)
            loss_clean = criterion(clean_logits, label)

            # adv loss
            adv_logits = detector(adv_dire)
            loss_adv = criterion(adv_logits, label)

            loss = (1 - adv_ratio) * loss_clean + adv_ratio * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            clean_correct += (torch.argmax(clean_logits, dim=1) == label).sum().item()
            adv_correct += (torch.argmax(adv_logits, dim=1) == label).sum().item()
            total_samples += label.shape[0]

        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        clean_acc = clean_correct / max(total_samples, 1)
        adv_acc = adv_correct / max(total_samples, 1)
        print(f"Epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  "
              f"clean_acc={clean_acc:.4f}  adv_acc={adv_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # ========== Validation ==========
        detector.eval()
        val_correct = 0
        val_adv_correct = 0
        val_total = 0

        val_sample_size = cfg.get('val_sample_size', None)
        if val_sample_size and val_sample_size < len(val_data):
            val_indices = torch.randperm(len(val_data))[:val_sample_size].tolist()
            val_subset = Subset(val_data, val_indices)
            val_loader_epoch = DataLoader(val_subset, batch_size=cfg['batch_size'])
        else:
            val_loader_epoch = val_loader

        for image, label in tqdm(val_loader_epoch, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            if image.shape[0] != cfg['batch_size']:
                continue
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.no_grad():
                dire, _ = compute_dire(image * 2 - 1, model, diffusion, cfg)
                logits = detector(dire)
                val_correct += (torch.argmax(logits, dim=1) == label).sum().item()

            # 用 BPDA 攻击验证鲁棒性
            adv = pgd_bpda_attack(
                image=image,
                true_label=label,
                target_label=1 - label,
                targeted=True,
                model=model,
                diffusion=diffusion,
                cfg=cfg,
                detector=detector,
                eps=cfg.get('atk_eps', 8/255),
                alpha=cfg.get('atk_alpha', 1/255),
                steps=cfg.get('atk_steps', 10),
            )
            with torch.no_grad():
                adv_dire, _ = compute_dire(adv * 2 - 1, model, diffusion, cfg)
                adv_logits = detector(adv_dire)
                val_adv_correct += (torch.argmax(adv_logits, dim=1) == label).sum().item()

            val_total += label.shape[0]

        val_acc = val_correct / max(val_total, 1)
        val_adv_acc = val_adv_correct / max(val_total, 1)
        print(f"  Val clean_acc={val_acc:.4f}  adv_acc={val_adv_acc:.4f}")

        # 保存最佳模型（以对抗准确率为指标）
        if val_adv_acc > best_val_acc:
            best_val_acc = val_adv_acc
            torch.save(detector.state_dict(), os.path.join(save_path, "best.pt"))
            print(f"  Saved best model (adv_acc={best_val_acc:.4f})")

        # 每个 epoch 都保存
        torch.save(detector.state_dict(),
                   os.path.join(save_path, f"epoch{epoch+1}.pt"))

    print(f"Adversarial training done. Best val adv_acc={best_val_acc:.4f}")
    print(f"Checkpoints saved to: {save_path}")


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
        attack_dire(cfg, device)
    elif cfg["todo"] == "adv_train":
        print("Running adversarial training with BPDA...")
        adv_train_dire(cfg, device)


if __name__ == "__main__":
    main()