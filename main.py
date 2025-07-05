import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import os
import sys

import torchattacks
from torchattacks import PGD
from autoattack import AutoAttack


import numpy as np
import cv2

from tqdm import tqdm

from scripts.argument import parser
import logging
from scripts.trainer import Trainer
from scripts.load_data import (
    load_single_dataset,
    load_fold,
    load_artifact,
    load_diffusion_forensics,
    load_GenImage,
    load_data_2_path,
    load_datasets,
)


def main(args):
    print("---------------start---------------")
    print(args)

    device = "cuda:" + str(args.device)

    if args.model == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(2048, 2)
        if args.load_path:
            load_path = args.load_path
            m_state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(m_state_dict)
        model = model.to(device)
    elif args.model == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Linear(768, 2) 
        if args.load_path:
            load_path = args.load_path
            m_state_dict = torch.load(load_path, map_location="cuda")
            model.load_state_dict(m_state_dict)
        model = model.to(device)

    if args.adv_mode==0 or args.adv_mode==1:
        atk = PGD(
            model,
            eps=args.atk_eps,
            alpha=args.atk_alpha,
            steps=args.atk_steps,
            random_start=True,
        )
        atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
        atk.set_device(device)
    elif args.adv_mode==2:
        atk=AutoAttack(model, norm='Linf', eps=args.atk_eps, device=device,version='custom',attacks_to_run=['apgd-ce', 'fab','square'])
 
    trainer = Trainer(args, atk)
    if args.load_path2:
        load_path = args.load_path2
        m_state_dict = torch.load(load_path, map_location="cuda")
        model.load_state_dict(m_state_dict)
        print("load2")

    if args.adv:
        print("adv:True")
    else:
        print("adv:False")

    if args.n:
        args.data_types=['n']*len(args.data_paths)
 

    if args.todo == "train":

        dataset_path = args.dataset

        train_transform = transforms.Compose(
            [
                # transforms.RandomRotation(20),  # 随机旋转角度
                # transforms.ColorJitter(brightness=0.1),  # 颜色亮度
                transforms.Resize([args.data_size,args.data_size]),   
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize([args.data_size,args.data_size]),  
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        if not args.data_types:
            train_data, val_data = load_fold(
                dataset_path, train_transform, val_transform
            )
        else:
            train_data, val_data = load_datasets(
                args.data_paths,
                args.data_types,
                train_transform,
                args.validation_split,
                args.ratio_list,
            )
        """from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
        train_data=ConcatDataset([train_data,val_data])
        from scripts.load_data import spilt_dataset
        train_data, val_data=spilt_dataset(train_data)"""

        print("train data:" + str(len(train_data)))
        print("val data:" + str(len(val_data)))

        train_loader = data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=not args.not_shuffle,
            num_workers=args.num_workers,
        )
        val_loader = data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=not args.not_shuffle,
            num_workers=args.num_workers,
        )

        if args.train_dataset2:
            print("using train_dataset2")
            train_path2 = args.train_dataset2
            train_data2 = load_single_dataset(train_path2, transform=train_transform)
            train_loader2 = data.DataLoader(
                train_data2,
                batch_size=args.batch_size,
                shuffle=not args.not_shuffle,
                num_workers=args.num_workers,
            )
            print("train data2:" + str(len(train_data2)))
        else:
            train_loader2 = None
        if args.val_dataset2:
            print("using val_dataset2")
            val_path2 = args.val_dataset2
            val_data2 = load_single_dataset(val_path2, transform=val_transform)
            val_loader2 = data.DataLoader(
                val_data2,
                batch_size=args.batch_size,
                shuffle=not args.not_shuffle,
                num_workers=args.num_workers,
            )
            print("val data:2" + str(len(val_data2)))
        else:
            val_loader2 = None

        trainer.set_dataloader(
            train_loader=train_loader,
            train_loader2=train_loader2,
            val_loader=val_loader,
            val_loader2=val_loader2,
        )
        trainer.train(model, args.adv)

    elif args.todo == "test":

        val_path = args.dataset

        val_transform = transforms.Compose(
            [
                transforms.Resize([args.data_size, args.data_size]),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        if not args.data_types:
            val_data = load_single_dataset(val_path, transform=val_transform)
        else:
            train_data, val_data = load_datasets(
                args.data_paths,
                args.data_types,
                val_transform,
                args.validation_split, 
            ) 
        print("val data:" + str(len(val_data)))
        val_loader = data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        if args.val_dataset2:
            print("using val_dataset2")
            val_path2 = args.val_dataset2
            val_data2 = load_single_dataset(val_path2, transform=val_transform)
            val_loader2 = data.DataLoader(
                val_data2,
                batch_size=args.batch_size,
                shuffle=not args.not_shuffle,
                num_workers=args.num_workers,
            )
        else:
            val_loader2 = None
        trainer.set_dataloader(val_loader=val_loader, val_loader2=val_loader2)
        trainer.evaluate(model, adv_test=args.adv or args.adv or args.diff_denoise)
    
    elif args.todo == "degrade":

        save_path = "checkpoint/" + args.save_path
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        int_files = [int(file) for file in os.listdir(save_path)]
        if len(int_files) == 0:
            save_path = os.path.join(save_path, "1")
        else:
            save_path = os.path.join(save_path, str(max(int_files) + 1))

        os.mkdir(save_path)
        trainer.set_loggers(
            save_path + "/" + args.save_path
        )  # 例：...savepath/2/savepath          顺序train,test,advtest

        val_path = args.dataset

        def my_evaluate(this_transform, namestr):
            val_data = datasets.ImageFolder(val_path, transform=this_transform)
            val_loader = data.DataLoader(
                val_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            print(namestr + " evaluate")
            trainer.evaluate(model, val_loader, adv_test=args.adv)

        val_transform = transforms.Compose(
            [ 
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(val_transform, "Normal")

        downsample_128 = transforms.Compose(
            [
                transforms.Resize(size=(128, 128)), 
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(downsample_128, "downsample_128")

        downsample_64 = transforms.Compose(
            [
                transforms.Resize(size=(64, 64)), 
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(downsample_64, "downsample_64")

        flip_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(flip_transform, "flip_transform")

        crop_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(224, 224), scale=(0.8, 0.85), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(crop_transform, "Crop")

        rotate_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=(-45, 45)),
                transforms.Resize([args.data_size, args.data_size]),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(rotate_transform, "Rotate")

    elif args.todo == "get_imgs":

        dataset_path = args.dataset
        save_path = args.save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        train_transform = transforms.Compose(
            [
                # transforms.RandomRotation(20),  # 随机旋转角度
                # transforms.ColorJitter(brightness=0.1),  # 颜色亮度
                transforms.Resize([args.data_size, args.data_size]),  # 设置成224×224大小的张量
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform=train_transform
        if not args.data_types:
            train_data, val_data = load_fold(
                dataset_path, train_transform=train_transform, val_transform=val_transform
            )
        else:
            train_data, val_data = load_datasets(
                args.data_paths,
                args.data_types,
                train_transform,
                args.validation_split,
                args.ratio_list,
            )

        data_loader = data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        save_path=args.save_path
        trainer.save_imgs(data_loader,save_path+'/train', args.adv, args.diff_denoise)

        data_loader = data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        trainer.save_imgs(data_loader,save_path+'/test', args.adv, args.diff_denoise)


if __name__ == "__main__":

    args = parser()

    main(args)
