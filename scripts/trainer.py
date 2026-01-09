import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import os
import sys

import torchattacks
from torchattacks import PGD

import numpy as np
import cv2

from tqdm import tqdm

from scripts.argument import parser
import logging

from scripts.load_data import (
    load_artifact,
    load_fold,
    load_diffusion_forensics,
    load_GenImage,
)
from scripts.loss import trades_loss, mart_loss
from scripts.utils import label_to_str, get_crt_num


class Trainer:
    def __init__(self, args, atk):
        self.args = args
        self.atk = atk
        self.device = "cuda:" + args.device
        self.loggers = []
        self.train_loader = None
        self.train_loader2 = None
        self.val_loader = None
        self.val_loader2 = None
        self.epoch = args.load_epoch

        if args.todo in ["train", "test"]:
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
            self.save_path = save_path  # 例：checkpoint/face_normal/3

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            self.set_loggers(
                self.save_path + "/" + args.save_path
            )  # 例：...savepath/2/savepath          顺序train,test,advtest


    def set_dataloader(
        self, train_loader=None, train_loader2=None, val_loader=None, val_loader2=None
    ):
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.val_loader = val_loader
        self.val_loader2 = val_loader2

    def train(
        self,
        model,
        adv_train=False,
    ):
        args = self.args

        criterion = torch.nn.CrossEntropyLoss()
        if args.sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.test_first:
            self.evaluate(
                model,
                adv_test=args.adv or args.adv_test,
            )
        for epoch in range(args.load_epoch, args.epoches):
            self.epoch = epoch
            train_loss, train_acc, losses = self.train_step(
                model,
                optimizer,
                criterion,
                adv_train=adv_train,
            )
            print(
                "epoch"
                + str(epoch + 1)
                + "  train_loss:"
                + str(train_loss)
                + "  train_acc:"
                + str(train_acc)
            )
            if adv_train:
                self.loggers[0].info(
                    f"Epoch{epoch}: Training accuracy: {train_acc[0]:.4f},{train_acc[1]:.4f}"
                )
            else:
                self.loggers[0].info(
                    f"Epoch{epoch}: Training accuracy: {train_acc[0]:.4f}"
                )
            if args.save_loss:
                np.save(
                    self.save_path + "/batch_losse_epoch" + str(epoch) + ".npy",
                    np.array(losses),
                )

            if (epoch + 1) % args.save_each_epoch == 0:
                self.evaluate(
                    model,
                    adv_test=args.adv_test or args.adv,
                )
                torch.save(
                    model.state_dict(), self.save_path + "/epoch" + str(epoch) + ".pt"
                )
            if args.adv and (epoch + 1) % args.update_adv_each_epoch == 0:
                print("reset atk")
                self.atk = PGD(
                    model,
                    eps=args.atk_eps,
                    alpha=args.atk_alpha,
                    steps=args.atk_steps,
                    random_start=True,
                )
                self.atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
                self.atk.set_device(self.device)
        torch.save(
            model.state_dict(),
            self.save_path + "/final_epoch" + str(args.epoches) + ".pt",
        )
        print(f"Save in: {self.save_path}")

    def train_step(
        self,
        model,
        optimizer,
        criterion,
        adv_train=False,
    ):
        args = self.args
        atk = self.atk
        device = self.device

        model.train()

        total_loss = 0
        train_corrects = 0
        adv_corrects = 0
        train_sum = 0

        if self.train_loader2:
            dataloader_iterator = iter(self.train_loader2)

        batch_count = 0
        losses = []
        acc = []
        for i, (image, label) in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            image = image.to(device)
            label = label.to(device)
            if self.train_loader2:
                try:
                    image2, label2 = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.train_loader2)
                    image2, label2 = next(dataloader_iterator)
                image2 = image2.to(device)
                label2 = label2.to(device)

                image = torch.cat([image, image2], dim=0)
                label = torch.cat([label, label2], dim=0)

            optimizer.zero_grad()
            target = model(image)
            loss = criterion(target, label)
            # loss=0
            if adv_train:
                adv_image = self.get_adv_imgs(
                    model,
                    x_natural=image,
                    y=label,
                    adv_mode=args.adv_mode,
                    mode="train",
                )
                if args.TRADES:
                    loss, adv_correct_num = trades_loss(
                        model=model,
                        x_natural=image,
                        y=label,
                        x_adv=adv_image,
                        optimizer=optimizer,
                        beta=args.TRADES_beta,
                    )
                elif args.MART:
                    loss, adv_correct_num = mart_loss(
                        model=model,
                        x_natural=image,
                        y=label,
                        x_adv=adv_image,
                        optimizer=optimizer,
                        beta=args.MART_beta,
                    )
                elif args.normal_adv:
                    target2 = model(adv_image)
                    loss_adv = criterion(target2, label)
                    loss = loss + loss_adv
                    adv_correct_num = get_crt_num(target2, label)

                adv_corrects += adv_correct_num

            loss.backward()
            optimizer.step()
            batch_count += 1

            total_loss += loss.item()
            train_corrects += get_crt_num(target, label)
            train_sum += label.shape[0]
            losses.append(loss.item())
            if not args.test_each_batch == 0:
                if (i + 1) % args.test_each_batch == 0:
                    # this_acc = np.sum(pred_label == true_label) / pred_label.shape[0]
                    test_loss, Scores = self.evaluate_step(
                        model,
                        self.val_loader,
                        criterion,
                        adv_test=False,
                        log_str=f"Batch_id:{i}",
                        logger_index=0,
                    )
                    if args.adv or args.adv_test:

                        test_loss, Scores = self.evaluate_step(
                            model,
                            self.val_loader,
                            criterion,
                            adv_test=True,
                            log_str=f"Batch_id:{i}  adv",
                            logger_index=0,
                        )
                        
                    model.train()
            # np.save("batch_losses.npy",np.array(losses))
        acc.append(train_corrects / train_sum)
        if adv_train:
            acc.append(adv_corrects / train_sum)
        return (
            total_loss / float(len(self.train_loader)),
            acc,
            losses,
        )

    def evaluate(self, model, adv_test=False):
        criterion = torch.nn.CrossEntropyLoss()

        args = self.args
        if adv_test:
            test_loss, Scores = self.evaluate_step(
                model,
                self.val_loader,
                criterion,
                adv_test=True,
                log_str="adv",
                logger_index=2,
            )
        else:
            test_loss, Scores = self.evaluate_step(
            model, self.val_loader, criterion, adv_test=False,log_str="val",logger_index=1
        )
        if self.val_loader2:
            test_loss, Scores = self.evaluate_step(
                model,
                self.val_loader2,
                criterion,
                adv_test=False,
                log_str="another_val",
                logger_index=1,
            )

    def evaluate_step(
        self,
        model,
        val_loader,
        criterion,
        adv_test=False,
        log_str="val",
        logger_index=1,
    ):
        device = self.device
        atk = self.atk
        args = self.args
        model.eval()
        corrects = eval_loss = 0
        test_sum = 0

        all_preds = []
        all_labels = []
        for image, label in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)
            if adv_test:
                image = self.get_adv_imgs(
                    model, x_natural=image, y=label, adv_mode=args.adv_mode, mode="val"
                )
                

            with torch.no_grad():
                pred = model(image)
                loss = criterion(pred, label)
                eval_loss += loss.item()
                max_value, max_index = torch.max(pred, 1)
                pred_label = max_index.cpu().numpy()
                true_label = label.cpu().numpy()
                all_preds.extend(pred_label)
                all_labels.extend(true_label)

        test_loss = eval_loss / float(len(val_loader))
        print(len(all_labels))
        print(len(all_preds))
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)


        print(
            f"Epoch{self.epoch}: {log_str} Loss:{test_loss}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, tn, fp, fn, tp : {tn}, {fp}, {fn}, {tp}"
        )
        self.loggers[logger_index].info(
            f"Epoch{self.epoch}: {log_str} Loss:{test_loss}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, tn, fp, fn, tp : {tn}, {fp}, {fn}, {tp}"
        )

        return test_loss, [tn, fp, fn, tp, accuracy, precision, recall, f1]

    def set_loggers(self, save_path):
        args = self.args
        self.loggers = []

        train_logger = self.get_logger(save_path, "train")
        test_logger = self.get_logger(save_path, "test")
        self.loggers.append(train_logger)
        self.loggers.append(test_logger)
        if args.adv or args.adv_test:
            adv_test_logger = self.get_logger(save_path, "adv_test")
            self.loggers.append(adv_test_logger)

    def get_logger(self, save_path, typestr):
        # 创建train和test日志记录器
        this_logger = logging.getLogger(typestr)
        this_logger.setLevel(logging.INFO)

        this_file_handler = logging.FileHandler(save_path + "_" + typestr + ".log")
        this_file_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        this_file_handler.setFormatter(formatter)

        # 将处理器添加到日志记录器
        this_logger.addHandler(this_file_handler)
        return this_logger

    def save_imgs(
        self,model, data_loader, save_path, adv=False, normalize=False
    ):
        device = self.device
        args = self.args
        atk = self.atk
        i = 0
        j = 0
        for image, label in tqdm(data_loader):
            image = image.to(device)
            label = label.to(device)
            
            if adv:
                imgs = self.get_adv_imgs(
                    model, x_natural=image, y=label, adv_mode=args.adv_mode, mode="val"
                )
            for t in range(len(label)):
                this_label = label[t].cpu().numpy().astype(np.uint8)
                label_str = label_to_str(this_label)
                os.makedirs(f"{save_path}/{label_str}", exist_ok=True)
                if this_label == 0:
                    i += 1
                    k = i
                else:
                    j += 1
                    k = j
                torchvision.utils.save_image(
                    imgs[t], f"{save_path}/{label_str}/{str(k)}.png"
                )

    def get_adv_imgs(
        self,
        model,
        x_natural,
        y,
        mode="train",
        adv_mode=0,
        step_size=0.003,
        epsilon=0.031,
        perturb_steps=10,
        distance="l_inf",
    ):
        model.eval()
        if adv_mode == 0:
            image = x_natural
            label = y
            x_adv = self.atk(image, label)
        elif adv_mode == 1:
            device = next(model.parameters()).device
            x_adv = (
                x_natural.detach()
                + 0.001 * torch.randn(x_natural.shape).to(device).detach()
            )
            if distance == "l_inf":
                for _ in range(perturb_steps):
                    x_adv.requires_grad_()
                    with torch.enable_grad():
                        loss_ce = F.cross_entropy(model(x_adv), y)
                    grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                    x_adv = torch.min(
                        torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
                    )
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
            else:
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif adv_mode == 2:
            x_adv = self.atk.run_standard_evaluation(
                x_natural, y, bs=self.args.batch_size
            )
        elif adv_mode == 3 or adv_mode == 4:
            x_adv = self.atk.attack(model, x_natural, labels=y, targeted=False)
        if mode == "train":
            model.train()
        return x_adv
