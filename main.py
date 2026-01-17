import os
import sys


import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torchattacks
from torchattacks import PGD
from autoattack import AutoAttack
from scripts.attacker import CWAttacker,SquareAttacker
from fast_adv.attacks import DDN

from scripts.argument import get_parser
from scripts.trainer import Trainer
from scripts.load_data import (
    load_single_dataset,
    load_fold,
    load_artifact,
    load_diffusion_forensics,
    load_GenImage,
    load_data_2_path,
    load_datasets,
    load_norm_data
)

def setup_model(args,device):
    if args.model == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(2048, 2)
    elif args.model == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Linear(768, 2)
        
    if args.load_path:
        load_path = args.load_path
        m_state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(m_state_dict)
    model = model.to(device, non_blocking=True)
    return model

def setup_attacker(args,model,device):
    if args.adv_mode == 0 or args.adv_mode == 1:
        atk = PGD(
            model,
            eps=args.atk_eps,
            alpha=args.atk_alpha,
            steps=args.atk_steps,
            random_start=True,
        )
        atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
        atk.set_device(device)
    elif args.adv_mode == 2:
        atk = AutoAttack(
            model,
            norm="Linf",
            eps=args.atk_eps,
            verbose=False,
            device=device,
            version="custom",
            attacks_to_run=["apgd-ce", "fab", "square"],
        )
    elif args.adv_mode == 3: 
        atk = DDN(steps=100, device=device)
    elif args.adv_mode == 4:
        atk = CWAttacker(model, device, c=args.CW_c)
    elif args.adv_mode == 5:
        atk = SquareAttacker(model, device,norm="l2")
    elif args.adv_mode == 6:
        atk = SquareAttacker(model,device,norm="linf")
    return atk

def prepare_transforms(args):
    train_transform = transforms.Compose(
        [
            # transforms.RandomRotation(20),  # 随机旋转角度
            # transforms.ColorJitter(brightness=0.1),  # 颜色亮度
            transforms.Resize([args.data_size, args.data_size]),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize([args.data_size, args.data_size]),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )
    return train_transform, val_transform

def prepare_dataloader(data_paths,train_transform,val_transform,combine_all=True,args=None):
    val_split=args.validation_split
    ratio_list=args.ratio_list
    b_s=args.batch_size
    shuffle=not args.not_shuffle
    n_w=args.num_workers
    val_rate=args.val_rate

    train_data, val_data = load_norm_data(
        data_paths,
        train_transform,
        val_transform,
        val_split,
        ratio_list,
        concat=combine_all
    )
    
    train_loaders=[]
    val_loaders = []
    for i in range(len(data_paths)):
        t_loader = data.DataLoader(
            train_data[i],
            batch_size=b_s,
            shuffle=shuffle,
            num_workers=n_w,
        )
        if val_rate<1.0:
            val_data.shrink_dataset(args.val_rate)
        v_loader = data.DataLoader(
            val_data[i],
            batch_size=b_s,
            shuffle=shuffle,
            num_workers=n_w,
        )
        train_loaders.append(t_loader)
        val_loaders.append(v_loader)
        
    return train_loaders, val_loaders
    
    
    
    
def run_train(args, trainer, model):
    train_transform, val_transform = prepare_transforms(args)

    train_loader1,val_loader1=prepare_dataloader(args.data_paths,train_transform,val_transform,combine_all=True,args=args)
    train_loader1=train_loader1[0]
    val_loader1=val_loader1[0]
    if args.data_paths2:
        train_loader2,val_loader2=prepare_dataloader(args.data_paths2,train_transform,val_transform,combine_all=True,args=args)
        train_loader2=train_loader2[0]
        val_loader2=val_loader2[0]
    else:
        train_loader2=None
        val_loader2=None
    
    trainer.set_dataloader(
        train_loader=train_loader1,
        train_loader2=train_loader2,
        val_loader=val_loader1,
        val_loader2=val_loader2,
    )
    trainer.train(model, args.adv)
    
    
    

def run_test(args,trainer,model):
    train_transform, val_transform = prepare_transforms(args)
    
    _,val_loader=prepare_dataloader(args.data_paths,val_transform,val_transform,combine_all=True,args=args)

    trainer.set_dataloader(val_loader=val_loader[0])
    trainer.evaluate(model, adv_test=args.adv or args.adv or args.diff_denoise)
    if args.not_combine:
        _,val_loaders=prepare_dataloader(args.data_paths,val_transform,val_transform,combine_all=False,args=args)
        for i in range(len(args.data_paths)):
            print(f"Testing on dataset: {args.data_paths[i].split('/')[-1]}")
            single_val_loader=val_loaders[i]
            trainer.set_dataloader(val_loader=single_val_loader)
            trainer.evaluate(model, adv_test=args.adv or args.adv or args.diff_denoise)
 
def run_get_imgs(args,trainer,model):
    train_transform, val_transform = prepare_transforms(args)
    save_path = args.save_path
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    train_loader,val_loader=prepare_dataloader(args.data_paths,val_transform,val_transform,combine_all=True,args=args)
    
    if not args.val_only:
        trainer.save_imgs(model,train_loader,save_path+'/train', args.adv, args.diff_denoise)
    trainer.save_imgs(model,val_loader,save_path+'/test', args.adv, args.diff_denoise)

def main(args):
    print("---------------start---------------")
    device = "cuda:" + str(args.device)
    print("using device:" + device)
    
    
    model = setup_model(args,device)

    if args.adv:
        atk = setup_attacker(args,model,device)
    else:
        atk=None
    
    trainer = Trainer(args, atk)
    
    if args.n:
        args.data_types = ["n"] * len(args.data_paths)
    
    task_map = {
        "train": run_train,
        "test": run_test,
        "get_imgs": run_get_imgs
    }
    
    if args.todo in task_map:
        task_map[args.todo](args, trainer, model)
    else:
        print(f"Unknown task: {args.todo}")
   

if __name__ == "__main__":

    args = get_parser()
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    