import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


class CWAttacker:
    def __init__(self, model, device, c=0.0001):
        """
        初始化CW攻击类
        :param model: 目标模型
        :param device: 使用的设备（如'cuda'或'cpu'）
        :param c: 攻击的强度系数，控制攻击的影响
        """
        self.model = model
        self.device = device
        self.c = c

    def cw_l2_attack(
        self,
        model,
        images,
        labels,
        targeted=False,
        c=1e-4,
        kappa=0,
        max_iter=1000,
        learning_rate=0.01,
    ):

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Define f-function
        def f(x):

            outputs = model(x)
            one_hot_labels = torch.eye(len(outputs[0])).to(self.device)[labels]

            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            # If targeted, optimize for making the other class most likely
            if targeted:
                return torch.clamp(i - j, min=-kappa)

            # If untargeted, optimize for making the other class most likely
            else:
                return torch.clamp(j - i, min=-kappa)

        w = torch.zeros_like(images, requires_grad=True).to(self.device)

        optimizer = optim.Adam([w], lr=learning_rate)

        prev = 1e10

        for step in range(max_iter):

            a = 1 / 2 * (nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction="sum")(a, images)
            loss2 = torch.sum(c * f(a))

            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (max_iter // 10) == 0:
                if cost > prev:
                    print("Attack Stopped due to CONVERGENCE....")
                    return a
                prev = cost

            print(
                "- Learning Progress : %2.2f %%        "
                % ((step + 1) / max_iter * 100),
                end="\r",
            )

        attack_images = 1 / 2 * (nn.Tanh()(w) + 1)

        return attack_images
    def attack(self,model, x_natural, labels, targeted=False):
        return  self.cw_l2_attack(model,images=x_natural,labels=labels,targeted=targeted,c=self.c)
    

class SquareAttacker:
    def __init__(self, model, device, norm,eps=0.1, n_iter=10000, p=0.05,loss_type='cross_entropy'):
        self.model = model
        self.device = device
        self.norm = norm
        self.eps = eps
        self.n_iter = n_iter
        self.p = p if self.norm == 'l2' else 0.05
        self.loss_type = loss_type
    def attack(self,model, x_natural, labels, targeted=False):        
        if self.norm == 'l2':
            return square_attack_l2(model, x_natural, labels, self.eps,self.n_iter, self.p, targeted, self.loss_type)
        elif self.norm == 'linf':
            return square_attack_linf(model, x_natural, labels,  self.eps,self.n_iter, self.p, targeted, self.loss_type)
        else:
            raise ValueError("Unsupported norm type: {}".format(self.norm))
def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 2