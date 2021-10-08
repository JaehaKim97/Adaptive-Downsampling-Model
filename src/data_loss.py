import torch
import torch.nn as nn
import torch.nn.functional as F
from bicubic_pytorch import core

def get_data_loss(img_s, img_gen, data_loss_type, down_filter, args):
    criterionL1 = nn.L1Loss()

    if data_loss_type == 'adl':
        if down_filter is not None:
            padL = args.adl_ksize // 2
            padR = args.adl_ksize // 2
            if args.adl_ksize % 2 == 0:
                padL -= 1
                
            filtered_img_s = down_filter(F.pad(img_s,(padL,padR,padL,padR),mode='replicate'))
            down_filtered_img_s = F.interpolate(filtered_img_s, scale_factor=0.5, mode='nearest',
                                                                recompute_scale_factor=False)
            return criterionL1(down_filtered_img_s, img_gen)
        else:
            # use lal for initial few epochs for stablizing
            return get_data_loss(img_s, img_gen, 'lfl', down_filter, args)

    elif data_loss_type == 'lfl':
        hr_filter = nn.AvgPool2d(kernel_size=args.box_size*2, stride=args.box_size*2)
        lr_filter = nn.AvgPool2d(kernel_size=args.box_size, stride=args.box_size)
        return criterionL1(lr_filter(img_gen), hr_filter(img_s))

    elif data_loss_type == 'bic':
        return criterionL1(core.imresize(img_s, scale=0.5), img_gen)

    elif data_loss_type == 'gau':
        gau_filter = GaussianLoss(scale=int(args.scale), sigma=args.gaussian_sigma,
                        kernel_size=args.gaussian_ksize, strided=(not args.gaussian_dense)).cuda()
        return gau_filter(img_gen, img_s)

    else:
        raise NotImplementedError('Not supported data loss type')



class GaussianLoss(nn.Module):
    def __init__(
            self,
            n_colors: int=3,
            kernel_size: int=16,
            scale: int=2,
            sigma: float=2.0,
            strided: bool=True,
            distance: str='l1') -> None:

        super().__init__()
        kx = gaussian_kernel(kernel_size=kernel_size, scale=1, sigma=sigma)
        kx = to_4d(kx, n_colors)
        self.register_buffer('kx', kx)

        ky = gaussian_kernel(
            kernel_size=(scale * kernel_size),
            scale=scale,
            sigma=sigma,
        )
        ky = to_4d(ky, n_colors)
        self.register_buffer('ky', ky)

        self.scale = scale
        self.strided = strided
        self.distance = distance

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = filter_loss(
            x,
            self.kx,
            y,
            self.ky,
            scale=self.scale,
            strided=self.strided,
            distance=self.distance,
        )
        return loss


def filter_loss(
        x: torch.Tensor,
        kx: torch.Tensor,
        y: torch.Tensor,
        ky: torch.Tensor,
        scale: int=2,
        strided: bool=True,
        distance: str='l1') -> torch.Tensor:

    wx = x.size(-1)
    wy = y.size(-1)
    # x should be smaller than y
    if wx >= wy:
        return filter_loss(y, ky, x, kx, strided=strided, scale=scale)

    if strided:
        sx = ky.size(-1)
    else:
        sx = 1

    sy = scale * sx
    x = F.conv2d(x, kx, stride=sx, padding=0)
    y = F.conv2d(y, ky, stride=sy, padding=0)

    if distance == 'l1':
        loss = F.l1_loss(x, y)
    elif distance == 'mse':
        loss = F.mse_loss(x, y)
    else:
        raise ValueError('{} loss is not supported!'.format(distance))

    return loss

def gaussian_kernel(
        kernel_size: int=16,
        scale: int=1,
        sigma: float=2.0) -> torch.Tensor:

    kernel_half = kernel_size // 2
    # Distance from the center point
    if kernel_size % 2 == 0:
        r = torch.linspace(-kernel_half + 0.5, kernel_half - 0.5, kernel_size)
    else:
        r = torch.linspace(-kernel_half, kernel_half, kernel_size)

    # Do not backpropagate through the kernel
    r.requires_grad = False
    r /= scale

    r = r.view(1, -1)
    r = r.repeat(kernel_size, 1)
    r = r ** 2
    r = r + r.t()

    exponent = -r / (2 * sigma**2)
    k = exponent.exp()
    k = k / k.sum()
    return k

def to_4d(k: torch.Tensor, n_colors: int) -> torch.Tensor:
    with torch.no_grad():
        k.unsqueeze_(0).unsqueeze_(0)
        k = k.repeat(n_colors, n_colors, 1, 1)
        e = torch.eye(n_colors, n_colors)
        e.unsqueeze_(-1).unsqueeze_(-1)
        k *= e

    return k