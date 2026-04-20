"""基准测试：64 张图片统计各阶段耗时"""
import time
import yaml
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from compute_dire.guided_diffusion.script_util import (
    model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict,
)
from compute_dire.guided_diffusion import dist_util
from scripts.attack_dire import compute_dire

import sys
sys.path.insert(0, ".")
# 复用 2main.py 里的 BPDA / PGD
import importlib.util
spec = importlib.util.spec_from_file_location("main2", "2main.py")
main2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main2)
pgd_bpda_attack = main2.pgd_bpda_attack


def bench(name, fn, warmup=2, n_imgs=64):
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[{name:35s}] total={elapsed:7.2f}s  per_image={elapsed/n_imgs*1000:8.2f} ms")
    return elapsed


def main():
    device = "cuda:0"
    with open("configs/adv_train_dire.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 加载 diffusion ----
    defaults = model_and_diffusion_defaults()
    defaults.update(cfg)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(defaults, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(defaults["model_path"], map_location="cpu"))
    if defaults["use_fp16"]:
        model.convert_to_fp16()
    model.to(device).eval()

    # ---- 加载 detector ----
    detector = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    detector.fc = torch.nn.Linear(2048, 2)
    if cfg.get("load_path"):
        detector.load_state_dict(torch.load(cfg["load_path"], map_location="cpu"))
    detector.to(device)

    # ---- 构造 64 张测试图片 ----
    N = 64
    BS = cfg.get("batch_size", 4)
    H = 256
    torch.manual_seed(0)
    imgs = torch.rand(N, 3, H, H, device=device)
    labels = torch.randint(0, 2, (N,), device=device)

    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def iter_batches():
        for i in range(0, N, BS):
            yield imgs[i:i+BS], labels[i:i+BS]

    # ===== 1. 推理 =====
    def eval_once():
        detector.eval()
        with torch.no_grad():
            for x, y in iter_batches():
                dire, _ = compute_dire(x * 2 - 1, model, diffusion, cfg)
                _ = detector(dire)
    t_eval = bench("推理 (eval: dire + detector)", eval_once, warmup=1)

    # ===== 2. Clean 训练 =====
    def clean_train_once():
        detector.train()
        for x, y in iter_batches():
            optimizer.zero_grad()
            with torch.no_grad():
                dire, _ = compute_dire(x * 2 - 1, model, diffusion, cfg)
            logits = detector(dire)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    t_clean = bench("Clean 训练 (dire + fwd + bwd)", clean_train_once, warmup=1)

    # ===== 3. BPDA 攻击 =====
    def attack_once():
        detector.eval()
        for x, y in iter_batches():
            _ = pgd_bpda_attack(
                image=x, true_label=y, target_label=1-y, targeted=True,
                model=model, diffusion=diffusion, cfg=cfg, detector=detector,
                eps=cfg.get("atk_eps", 8/255),
                alpha=cfg.get("atk_alpha", 1/255),
                steps=cfg.get("atk_steps", 10),
            )
    t_attack = bench("BPDA 攻击 (PGD steps=10)", attack_once, warmup=0)

    # ===== 4. 对抗训练 =====
    def adv_train_once():
        for x, y in iter_batches():
            # attack (detector.eval 内部)
            adv = pgd_bpda_attack(
                image=x, true_label=y, target_label=1-y, targeted=True,
                model=model, diffusion=diffusion, cfg=cfg, detector=detector,
                eps=cfg.get("atk_eps", 8/255),
                alpha=cfg.get("atk_alpha", 1/255),
                steps=cfg.get("atk_steps", 10),
            )
            detector.train()
            optimizer.zero_grad()
            with torch.no_grad():
                dire, _ = compute_dire(x * 2 - 1, model, diffusion, cfg)
                adv_dire, _ = compute_dire(adv * 2 - 1, model, diffusion, cfg)
            loss = criterion(detector(dire), y) + criterion(detector(adv_dire), y)
            loss.backward()
            optimizer.step()
    t_adv = bench("对抗训练 (attack + 2×dire + bwd)", adv_train_once, warmup=0)

    # ===== 汇总 =====
    print("\n" + "="*70)
    print(f"数据集规模: celeba(train)=162,079 + sfhq(train)=340,206 = 502,285 张")
    print("="*70)

    def fmt(seconds):
        h = seconds / 3600
        if h < 24:
            return f"{h:.1f} h"
        return f"{h/24:.1f} d"

    per_img = {
        "推理": t_eval / N,
        "Clean训练": t_clean / N,
        "BPDA攻击": t_attack / N,
        "对抗训练": t_adv / N,
    }

    epoch_sample = cfg.get("epoch_sample_size", 2000)
    val_sample = cfg.get("val_sample_size", 40)
    epochs = cfg.get("epochs", 20)

    print(f"\n使用配置: epochs={epochs}, epoch_sample_size={epoch_sample}, val_sample_size={val_sample}")
    print(f"\n{'阶段':<15s} {'每张耗时':>12s} {'采样集耗时':>14s} {'整个训练集耗时':>18s}")
    print("-"*70)
    for k, v in per_img.items():
        sample_time = v * epoch_sample
        full_time = v * 502285
        print(f"{k:<15s} {v*1000:>10.1f}ms {fmt(sample_time):>14s} {fmt(full_time):>18s}")

    # ===== 完整训练方案预估 =====
    print("\n" + "="*70)
    print("完整训练方案预估（每 epoch 采样 2000 张 + 验证 40 张）:")
    print("="*70)
    per_epoch = per_img["对抗训练"] * epoch_sample + (per_img["推理"] + per_img["BPDA攻击"]) * val_sample
    print(f"  单 epoch 耗时: {fmt(per_epoch)}")
    print(f"  {epochs} epochs 总耗时: {fmt(per_epoch * epochs)}")


if __name__ == "__main__":
    main()
