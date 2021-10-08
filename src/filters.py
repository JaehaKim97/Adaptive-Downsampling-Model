import math
import torch
from torch.nn import functional as F

def gaussian_kernel(sigma: float=1) -> torch.Tensor:
    kernel_width = math.ceil(3 * sigma)
    r = torch.linspace(-kernel_width, kernel_width, 2 * kernel_width + 1)
    r = r.view(1, -1)
    r = r.repeat(2 * kernel_width + 1, 1)
    r = r**2
    # Squared distance from origin
    r = r + r.t()

    exp = -r / (2 * sigma**2)
    coeff = exp.exp()
    coeff = coeff / coeff.sum()
    return coeff

def filtering(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = k.to(x.device)
    kh, kw = k.size()
    if x.dim() == 4:
        c = x.size(1)
        k = k.view(1, 1, kh, kw)
        k = k.repeat(c, c, 1, 1)
        e = torch.eye(c, c)
        e = e.to(x.device)
        e = e.view(c, c, 1, 1)
        k *= e
    else:
        raise ValueError('x.dim() == {}! It should be 3 or 4.'.format(x.dim()))

    x = F.pad(x, (kh // 2, kh // 2, kw // 2, kw // 2), mode='replicate')
    y = F.conv2d(x, k, padding=0)
    return y

def gaussian_filtering(x: torch.Tensor, sigma: float=1) -> torch.Tensor:
    k = gaussian_kernel(sigma=sigma)
    y = filtering(x, k)
    return y

def find_kernel(
        x: torch.Tensor,
        y: torch.Tensor,
        scale: int,
        k: int,
        max_patches: int=None,
        threshold: float=1e-5) -> torch.Tensor:
    '''
    Args:
        x (torch.Tensor): (B x C x H x W or C x H x W) A high-resolution image.
        y (torch.Tensor): (B x C x H x W or C x H x W) A low-resolution image.
        scale (int): Downsampling scale.
        k (int): Kernel size.
        max_patches (int, optional): Maximum number of patches to use.
            If not specified, use minimum number of patches.
            If set to -1, use all possible patches.
            You will get a better result with more patches.

        threshold (float, optional): Ignore values smaller than the threshold.

    Return:
        torch.Tensor: (k x k) The calculated kernel.
    '''
    if x.dim() == 3:
        x = x.unsqueeze(0)

    if y.dim() == 3:
        y = y.unsqueeze(0)

    bx, cx, hx, wx = x.size()
    by, cy, hy, wy = y.size()

    # If y is larger than x
    if hx < hy:
        return find_kernel(y, x)

    # We convert RGB images to grayscale
    def luminance(rgb):
        coeff = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        l = torch.sum(coeff * rgb, dim=1, keepdim=True)
        return l

    if cx == 3:
        x = luminance(x)
    if cy == 3:
        y = luminance(y)

    k_half = k // 2
    crop_y = math.ceil((k_half - 1) / 2)
    if crop_y > 0:
        y = y[..., crop_y:-crop_y, crop_y:-crop_y]
        hy_crop = hy - 2 * crop_y
        wy_crop = wy - 2 * crop_y

    # Flatten
    y = y.reshape(by, -1)

    hx_target = k + scale * (hy_crop - 1)
    crop_x = (hx - hx_target) // 2
    if crop_x > 0:
        x = x[..., crop_x:-crop_x, crop_x:-crop_x]
        hx_crop = hx - 2 * crop_x
        wx_crop = wx - 2 * crop_x

    x = F.unfold(x, k, stride=scale)
    x_spatial = x.view(bx, k, k, -1)

    '''
    Gradient-based sampling
    Caculate the gradient to determine which patches to use
    '''
    gx = x.new_zeros(1, k, k, 1)
    gx[:, k_half - 1, k_half - 1, :] = -1
    gx[:, k_half - 1, k_half, :] = 1
    grad_x = x_spatial * gx
    grad_x = grad_x.view(bx, k * k, -1)

    gy = x.new_zeros(1, k, k, 1)
    gy[:, k_half - 1, k_half - 1, :] = -1
    gy[:, k_half, k_half - 1, :] = 1
    grad_y = x_spatial * gy
    grad_y = grad_y.view(by, k * k, -1)

    grad = grad_x.sum(1).pow(2) + grad_y.sum(1).pow(2)
    grad_order = grad.view(-1).argsort(dim=-1, descending=True)

    # We need at least k^2 patches
    if max_patches is None:
        max_patches = k**2
    elif max_patches == -1:
        max_patches = len(grad_order)
    else:
        max_patches = min(max(k**2, max_patches), len(grad_order))

    grad_order = grad_order[:max_patches].view(-1)
    '''
    Increase precision for numerical accuracy.
    You will get wrong results with FLOAT32!!!
    '''
    # We use only one sample in the given batch
    x_sampled = x[0, ..., grad_order].double()
    x_t = x_sampled.t()

    y_sampled = y[0, ..., grad_order].double()
    y = y_sampled.unsqueeze(0)

    kernel = torch.matmul(y, x_t)
    kernel_c = torch.matmul(x_sampled, x_t)
    kernel_c = torch.inverse(kernel_c)
    kernel = torch.matmul(kernel, kernel_c)
    # For debugging
    #from scipy import io
    #io.savemat('tensor.mat', {'x_t': x_t.numpy(), 'y': y.numpy(), 'kernel': kernel.numpy()})

    # Kernel thresholding and normalization
    kernel = kernel * (kernel.abs() > threshold).double()
    #kernel = kernel / kernel.sum()
    kernel = kernel.view(k, k).float()
    return kernel

'''
if __name__ == '__main__':
    import numpy as np
    import imageio

    a = imageio.imread('../../lab/baby.png')
    a = np.transpose(a, (2, 0, 1))
    a = torch.from_numpy(a).unsqueeze(0).float()
    b = gaussian_filtering(a, sigma=0.3)
    b = b.round().clamp(min=0, max=255).byte()
    b = b.squeeze(0)
    b = b.numpy()
    b = np.transpose(b, (1, 2, 0))
    imageio.imwrite('../../lab/baby_filtered.png', b)

    #x = torch.arange(64).view(1, 1, 8, 8).float()
    #y = torch.arange(16).view(1, 1, 4, 4).float()
    from PIL import Image
    from torchvision.transforms import functional as TF
    x = Image.open('../../../dataset/DIV2K/DIV2K_train_HR/0001.png')
    x = TF.to_tensor(x)
    y = Image.open('DIV2K_train_LR_d104/X2/0001x2.png')
    y = TF.to_tensor(y)
    k = 20
    kernel = find_kernel(x, y, scale=2, k=k, max_patches=-1)

    kernel /= kernel.abs().max()
    k_pos = kernel * (kernel > 0).float()
    k_neg = kernel * (kernel < 0).float()
    k_rgb = torch.stack([-k_neg, k_pos, torch.zeros_like(k_pos)], dim=0)
    pil = TF.to_pil_image(k_rgb.cpu())
    pil = pil.resize((k * 20, k * 20), resample=Image.NEAREST)
    pil.save('kernel.png')
    pil.show()
'''

