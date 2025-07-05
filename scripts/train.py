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

import numpy as np
import cv2

from tqdm import tqdm

from argument import parser
import logging

from load_data import load_artifact, load_fold, load_diffusion_forensics, load_GenImage


class Trainer:
    def __init__(self, args, atk):
        self.args = args
        self.atk = atk
        self.device = "cuda:" + args.device
        self.loggers = []

    def train(
        self,
        model,
        train_loader,
        val_loader,
        adv_train=False,
        train_loader2=None,
        val_loader2=None,
    ):
        args = self.args
        atk = self.atk
        criterion = torch.nn.CrossEntropyLoss()
        if args.sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        save_path = "checkpoint/" + args.save_path
        int_files = [int(file) for file in os.listdir(save_path)]
        if len(int_files) == 0:
            save_path = os.path.join(save_path, "1")
        else:
            save_path = os.path.join(save_path, str(max(int_files) + 1))

        os.mkdir(save_path)
        self.set_loggers(
            save_path + "/" + args.save_path
        )  # 例：...savepath/2/savepath          顺序train,test,advtest

        self.loggers[0].info(f"Save in: {save_path}")

        if args.test_first:
            self.evaluate(
                model,
                val_loader,
                epoch=0,
                adv_test=args.adv or args.adv_test,
                val_loader2=val_loader2,
            )
        for epoch in range(args.load_epoch, args.epoches):
            train_loss, train_acc, losses = self.train_step(
                model,
                train_loader,
                optimizer,
                criterion,
                adv_train=adv_train,
                train_loader2=train_loader2,
                val_loader=val_loader,
            )
            print(
                "epoch"
                + str(epoch + 1)
                + "  train_loss:"
                + str(train_loss)
                + "  train_acc:"
                + str(train_acc)
            )
            self.loggers[0].info(f"Epoch{epoch}: Training accuracy: {train_acc:.4f}")
            np.save(
                save_path + "/epoch" + str(epoch) + "_batch_losses.npy",
                np.array(losses),
            )

            if (epoch + 1) % args.save_each_epoch == 0:
                self.evaluate(
                    model,
                    val_loader,
                    epoch=epoch + 1,
                    adv_test=args.adv_test or args.adv,
                    val_loader2=val_loader2,
                )
                torch.save(
                    model.state_dict(), save_path + "/epoch" + str(epoch + 1) + ".pt"
                )
            if args.adv and (epoch + 1) % args.update_adv_each_epoch == 0:
                self.atk = PGD(
                    model,
                    eps=args.atk_eps,
                    alpha=args.atk_alpha,
                    steps=args.atk_steps,
                    random_start=True,
                )
                self.atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
                self.atk.set_device(self.device)
        """self.evaluate(
            model,
            val_loader,
            epoch=args.epoches,
            adv_test=args.adv_test or args.adv,
            val_loader2=val_loader2,
        )"""
        torch.save(
            model.state_dict(), save_path + "/final_epoch" + str(args.epoches) + ".pt"
        )

    def train_step(
        self,
        model,
        train_loader,
        optimizer,
        criterion,
        adv_train=False,
        train_loader2=None,
        val_loader=None,
    ):
        args = self.args
        atk = self.atk
        device = self.device

        model.train()

        total_loss = 0
        train_corrects = 0
        train_sum = 0

        if train_loader2:
            dataloader_iterator = iter(train_loader2)

        batch_count = 0
        losses = []
        for i, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(device)
            label = label.to(device)
            if train_loader2:
                try:
                    image2, label2 = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(train_loader2)
                    image2, label2 = next(dataloader_iterator)
                image2 = image2.to(device)
                label2 = label2.to(device)

                image = torch.cat([image, image2], dim=0)
                label = torch.cat([label, label2], dim=0)

            optimizer.zero_grad()
            target = model(image)
            loss = criterion(target, label)
            print("Sdsdsd")
            print(image.shape)
            print(label.shape)
            if adv_train:
                adv_image = atk(image, label)
                target2 = model(adv_image)
                loss = loss + criterion(target2, label)

            loss.backward()
            optimizer.step()
            batch_count += 1

            total_loss += loss.item()
            max_value, max_index = torch.max(target, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            train_corrects += np.sum(pred_label == true_label)
            train_sum += pred_label.shape[0]
            losses.append(loss.item())
            if not args.test_each_batch == 0:
                if (i + 1) % args.test_each_batch == 0:
                    this_acc = np.sum(pred_label == true_label) / pred_label.shape[0]
                    test_loss, d, test_acc = self.evaluate_step(
                        model, val_loader, criterion, adv_test=False
                    )
                    self.loggers[0].info(
                        f"               Batch_id:{i} Batch Loss:{loss.item()} This acc: {this_acc} Normal Evaluate accuracy: {test_acc:.4f}"
                    )
                    if args.adv or args.adv_test:
                        test_loss, d, test_acc = self.evaluate_step(
                            model, val_loader, criterion, adv_test=True
                        )
                        self.loggers[0].info(
                            f"               Batch_id:{i} Batch Loss:{loss.item()} Adv Evaluate accuracy: {test_acc:.4f}"
                        )
                    model.train()
            # np.save("batch_losses.npy",np.array(losses))
        return total_loss / float(len(train_loader)), train_corrects / train_sum, losses

    def evaluate(self, model, val_loader, epoch=0, adv_test=False, val_loader2=None):
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, d, test_acc = self.evaluate_step(model, val_loader, criterion)
        print("val_loss:" + str(test_loss) + "  val_acc:" + str(test_acc))
        self.loggers[1].info(
            f"Epoch{epoch}: Loss:{test_loss} Evaluate accuracy: {test_acc:.4f}"
        )
        if adv_test:
            test_loss, d, test_acc = self.evaluate_step(
                model, val_loader, criterion, adv_test=True
            )
            print("adv_val_loss:" + str(test_loss) + "  adv_val_acc:" + str(test_acc))
            self.loggers[2].info(
                f"Epoch{epoch}: Adv Loss:{test_loss} Adv evaluate accuracy: {test_acc:.4f}"
            )
        if val_loader2:
            test_loss, d, test_acc = self.evaluate_step(
                model, val_loader2, criterion, adv_test=False
            )
            print(
                "another_val_loss:"
                + str(test_loss)
                + "  another_val_acc:"
                + str(test_acc)
            )
            self.loggers[1].info(
                f"Epoch{epoch}: Loss:{test_loss} Another Evaluate accuracy: {test_acc:.4f}"
            )

    def evaluate_step(self, model, val_loader, criterion, adv_test=False):
        device = self.device
        atk = self.atk
        model.eval()
        corrects = eval_loss = 0
        test_sum = 0
        for image, label in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)
            if adv_test:
                image = atk(image, label)
            with torch.no_grad():
                pred = model(image)
                loss = criterion(pred, label)
                eval_loss += loss.item()
                max_value, max_index = torch.max(pred, 1)
                pred_label = max_index.cpu().numpy()
                true_label = label.cpu().numpy()
                corrects += np.sum(pred_label == true_label)
                test_sum += pred_label.shape[0]
        return eval_loss / float(len(val_loader)), corrects, corrects / test_sum

    def set_loggers(self, save_path):
        args = self.args
        self.loggers = []

        train_logger = self.get_logger(save_path, "train")
        test_logger = self.get_logger(save_path, "test")
        self.loggers.append(train_logger)
        self.loggers.append(test_logger)
        if args.adv:
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

    def get_adv_imgs(self, data_loader):
        device = self.device
        args = self.args
        save_path = args.save_path
        atk = self.atk
        i = 0
        j = 0
        for image, label in tqdm(data_loader):
            image = image.to(device)
            label = label.to(device)
            imgs = atk(image, label)

            for t in range(len(label)):
                this_label = label[t].cpu().numpy().astype(np.uint8)
                os.makedirs(f"{save_path}/{str(this_label)}", exist_ok=True)
                if this_label == 0:
                    i += 1
                    k = i
                else:
                    j += 1
                    k = j
                torchvision.utils.save_image(
                    imgs[t], f"{save_path}/{str(this_label)}/{str(k)}.png"
                )


def main(args):
    print(args)

    batch_size = args.batch_size
    device = "cuda:" + str(args.device)

    if args.model == "resnet":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 2)
        if args.load_path:
            load_path = args.load_path
            m_state_dict = torch.load(load_path, map_location="cuda")
            model.load_state_dict(m_state_dict)
        model = model.to(device)
    elif args.model == "vit":
        model = models.vit_b_16(pretrained=True)
        model.heads = torch.nn.Linear(768, 2)
        if args.load_path:
            load_path = args.load_path
            m_state_dict = torch.load(load_path, map_location="cuda")
            model.load_state_dict(m_state_dict)
        model = model.to(device)

    atk = PGD(
        model,
        eps=args.atk_eps,
        alpha=args.atk_alpha,
        steps=args.atk_steps,
        random_start=True,
    )
    atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
    atk.set_device(device)

    trainer = Trainer(args, atk)

    if args.adv:
        print("adv:True")
    else:
        print("adv:False")

    if args.todo == "train":

        dataset_path = args.dataset
        train_path = dataset_path + "/train"
        val_path = dataset_path + "/test"

        save_path = "checkpoint/" + args.save_path
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        train_transform = transforms.Compose(
            [
                # transforms.RandomRotation(20),  # 随机旋转角度
                # transforms.ColorJitter(brightness=0.1),  # 颜色亮度
                transforms.Resize([224, 224]),  # 设置成224×224大小的张量
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )

        if args.artifact:
            train_data, val_data = load_artifact(
                dataset_path, train_transform, val_transform
            )
        elif args.df:
            train_data, val_data = load_diffusion_forensics(
                dataset_path, train_transform, val_transform
            )
        elif args.genimage:
            train_data, val_data = load_GenImage(
                dataset_path, args.imagenet, train_transform, val_transform
            )
        else:
            train_data, val_data = load_fold(
                dataset_path, train_transform, val_transform
            )

        train_loader = data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )
        val_loader = data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )

        if args.train_dataset2:
            print("using train_dataset2")
            train_path2 = args.train_dataset2
            train_data2 = datasets.ImageFolder(train_path2, transform=train_transform)
            train_loader2 = data.DataLoader(
                train_data2,
                batch_size=batch_size,
                shuffle=args.shuffle,
                num_workers=args.num_workers,
            )
        else:
            train_loader2 = None
        if args.val_dataset2:
            print("using val_dataset2")
            val_path2 = args.val_dataset2
            val_data2 = datasets.ImageFolder(val_path2, transform=val_transform)
            val_loader2 = data.DataLoader(
                val_data2,
                batch_size=batch_size,
                shuffle=args.shuffle,
                num_workers=args.num_workers,
            )
        else:
            val_loader2 = None
        trainer.train(
            model, train_loader, val_loader, args.adv, train_loader2, val_loader2
        )

    elif args.todo == "test":

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

        val_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )

        val_data = datasets.ImageFolder(val_path, transform=val_transform)
        val_loader = data.DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
        )

        trainer.evaluate(model, val_loader, adv_test=args.adv)
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
                batch_size=batch_size,
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
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        my_evaluate(rotate_transform, "Rotate")

    elif args.todo == "get_adv_imgs":

        dataset_path = args.dataset
        save_path = args.save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ]
        )
        imgdata = datasets.ImageFolder(dataset_path, transform=transform)
        data_loader = data.DataLoader(
            imgdata, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
        )

        trainer.get_adv_imgs(data_loader)


if __name__ == "__main__":

    args = parser()

    main(args)
