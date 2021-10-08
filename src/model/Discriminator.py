import torch.nn as nn
import torch
import torch.nn.functional as F

def SpectralConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
  return spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class Discriminator(nn.Module):
    def __init__(self, args, num_conv_block=4):
        super(Discriminator, self).__init__()
        block = []

        in_channels = 3
        out_channels = 64

        ConvKind = nn.Conv2d
        #if args.dis_spectral_norm:
        #    ConvKind = SpectralConv2d

        for _ in range(num_conv_block):
            block += [nn.ReflectionPad2d(1),
                      ConvKind(in_channels, out_channels, 3),
                      nn.LeakyReLU(),
                      nn.BatchNorm2d(out_channels)]
            in_channels = out_channels

            block += [nn.ReflectionPad2d(1),
                      ConvKind(in_channels, out_channels, 3, 2),
                      nn.LeakyReLU()]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [ConvKind(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  ConvKind(out_channels, out_channels, 3)]

        self.feature_extraction = nn.Sequential(*block)

        self.classification = nn.Sequential(
            nn.Linear(8192, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))