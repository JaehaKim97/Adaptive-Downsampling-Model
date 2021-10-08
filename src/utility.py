import os
import torch
import torch.nn as nn
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random


class log_writer():
    def __init__(self, experiment_dir, name):
        self.log_txt = os.path.join(os.path.join(experiment_dir, name),'training_log.txt')
        f = open(self.log_txt, 'w')

    def write(self, log):
        print(log)
        f = open(self.log_txt, 'a')
        f.write(log+'\n')


def plot_loss_down(save, loss_d, loss_g, loss_dl):
    assert len(loss_d) == len(loss_g)
    assert len(loss_d) == len(loss_dl)
    axis = np.linspace(1, len(loss_d), len(loss_d))
    fig = plt.figure()

    plt.plot(axis, loss_d, axis, loss_g, axis, loss_dl)

    #plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss_d(blue), loss_g(orange), loss_dl(green)')
    plt.grid(True)
    plt.savefig(os.path.join(save,'down_loss_graph.pdf'))
    plt.close(fig)

def plot_psnr(save, psnrs):
    axis = np.linspace(1, len(psnrs), len(psnrs))
    fig = plt.figure()

    plt.plot(axis, psnrs)

    #plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(save,'sr_psnr_graph.pdf'))
    plt.close(fig)


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def get_gaussian_kernel(kernel_size=5, sigma=1, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                                        torch.exp(
                                                -torch.sum((xy_grid - mean)**2., dim=-1) /\
                                                (2*variance)
                                        )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                                            kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

def get_avgpool_kernel(kernel_size=16, stride=1, channels=3):
    my_zeros = torch.empty(kernel_size, kernel_size)
    my_kernel = torch.ones_like(my_zeros)
    my_kernel = my_kernel / torch.sum(my_kernel)

    # Reshape to 2d depthwise convolutional weight
    my_kernel = my_kernel.view(1, 1, kernel_size, kernel_size)
    my_kernel = my_kernel.repeat(channels, 1, 1, 1)

    my_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, groups=channels, stride=stride, bias=False)

    my_filter.weight.data = my_kernel
    my_filter.weight.requires_grad = False
    
    return my_filter


def calc_psnr(sr, hr, scale, rgb_range=1, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if False: #dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = int(scale) + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def quantize(img, rgb_range=1, fake=False):
    pixel_range = 255 / rgb_range
    if fake:
        fake_img = img.mul(pixel_range).clamp(0, 255).round().div(pixel_range).detach()
        res = fake_img - img.detach()
        return img + res
    else:
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def _normalize(*args, mul=0.5, add=0.5, reverse=False):

    if reverse:
        ret = [
            (args[0] + add) * mul,
            *[(a + add) * mul for a in args[1:]]
        ]
    else:
        ret = [
            args[0] * mul + add,
            *[a * mul + add for a in args[1:]]
        ]

    return ret


if __name__ == '__main__':
        torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
        k = get_avgpool_kernel(kernel_size=16)
