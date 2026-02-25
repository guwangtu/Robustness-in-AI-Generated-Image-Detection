import torch
import numpy as np
import torchvision


class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t

    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1

        x = x * 2 - 1

        indices = list(range(t + 1))[::-1]

        t = torch.full((x.shape[0],), t).long().to(x.device)

        x_t = self.diffusion.q_sample(x, t)

        sample = x_t

        # print(x_t.min(), x_t.max())

        # si(x_t, 'vis/noised_x.png', to_01=True)

        # visualize
        l_sample = []
        l_predxstart = []

        for i in indices:

            out = self.diffusion.ddim_sample(
                self.model, sample, torch.full((x.shape[0],), i).long().to(x.device)
            )

            sample = out["sample"]

            l_sample.append(out["sample"])
            l_predxstart.append(out["pred_xstart"])

        # visualize
        si(torch.cat(l_sample), "l_sample.png", to_01=1)
        si(torch.cat(l_predxstart), "l_pxstart.png", to_01=1)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def forward(self, x):

        out = self.sdedit(x, self.t)  # [0, 1]
        out = self.classifier(out)
        return out


def si(x, p, to_01=False, normalize=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if to_01:
        torchvision.utils.save_image((x + 1) / 2, p, normalize=normalize)
    else:
        torchvision.utils.save_image(x, p, normalize=normalize)


def purify(x, t, model, diffusion):
    p_net = Denoised_Classifier(diffusion, model, classifier=None, t=t)
    return p_net.sdedit(x, t)
