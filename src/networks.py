import    torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as py_models
import numpy
import copy


####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class D_Module(nn.Module):
    def __init__(self, args, n_layer=5, norm='None', sn=False):
        super(D_Module, self).__init__()
        ch = 64

        self.args = args
        self.Diss = nn.ModuleList()
        self.Diss.append(self._make_net(ch, 3, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = [MyConv2d(input_dim, ch, kernel_size=7, stride=1, padding=3, norm=norm, sn=sn, Leaky=True)]
        tch = ch
        for _ in range(1,n_layer):
            model += [MyConv2d(tch, min(1024, tch * 2), kernel_size=5, stride=2, padding=2, norm=norm, sn=sn, Leaky=True)]
            tch *= 2
            tch = min(1024, tch)
        model += [nn.Conv2d(tch, 1, 2, 1, 0)]

        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        outs.append(self.Diss[0](x))
        
        return outs


####################################################################
#--------------------------- Generators ----------------------------
####################################################################

class G_Module(nn.Module):
    def __init__(self, args, norm=None, nl_layer=None):
        super(G_Module, self).__init__()
        
        tch = 64
        res = True
        
        headB = [MyConv2d(3, tch, kernel_size=3, stride=1, padding=1, norm=norm)]
        self.headB = nn.Sequential(*headB)

        bodyB1 = [MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=res),
                            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=res),
                            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=res),
                            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=res),]

        bodyB2 = [MyConv2d(tch, tch*2, kernel_size=2, stride=2, padding=0, norm=norm),]
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
    
        bodyB3 = [MyConv2d(tch*2, tch*2, kernel_size=3, stride=1, padding=1, norm=norm, Res=res),
                            MyConv2d(tch*2, tch*2, kernel_size=3, stride=1, padding=1, norm=norm, Res=res),
                            MyConv2d(tch*2, tch*2, kernel_size=3, stride=1, padding=1, norm=norm, Res=res), ]

        self.bodyB1 = nn.Sequential(*bodyB1)
        self.bodyB2 = nn.Sequential(*bodyB2)
        self.bodyB3 = nn.Sequential(*bodyB3)

        tailB = [ nn.Conv2d(tch*2, 3, kernel_size=1, stride=1, padding=0) ]
        self.tailB = nn.Sequential(*tailB)
 

    def forward(self, HR):
        tres = self.avgpool2(HR)
        out = self.headB(HR)
        
        res = out
        out = self.bodyB1(out)
        out += res

        out = self.bodyB2(out)

        res = out
        out = self.bodyB3(out)
        out += res

        out = self.tailB(out)
        out += tres

        return out

####################################################################
#--------------------------- losses ----------------------------
####################################################################
class PerceptualLoss():
        def __init__(self, loss, gpu=0, p_layer=14):
                super(PerceptualLoss, self).__init__()
                self.criterion = loss
                
                cnn = py_models.vgg19(pretrained=True).features
                cnn = cnn.cuda()
                model = nn.Sequential()
                model = model.cuda()
                for i,layer in enumerate(list(cnn)):
                        model.add_module(str(i),layer)
                        if i == p_layer:
                                break
                self.contentFunc = model         

        def getloss(self, fakeIm, realIm):
                if isinstance(fakeIm, numpy.ndarray):
                        fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
                        realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
                f_fake = self.contentFunc.forward(fakeIm)
                f_real = self.contentFunc.forward(realIm)
                f_real_no_grad = f_real.detach()
                loss = self.criterion(f_fake, f_real_no_grad)
                return loss

class PerceptualLoss16():
        def __init__(self, loss, gpu=0, p_layer=14):
                super(PerceptualLoss16, self).__init__()
                self.criterion = loss
#                 conv_3_3_layer = 14
                checkpoint = torch.load('/vggface_path/VGGFace16.pth')
                vgg16 = py_models.vgg16(num_classes=2622)
                vgg16.load_state_dict(checkpoint['state_dict'])
                cnn = vgg16.features
                cnn = cnn.cuda()
#                 cnn = cnn.to(gpu)
                model = nn.Sequential()
                model = model.cuda()
                for i,layer in enumerate(list(cnn)):
#                         print(layer)
                        model.add_module(str(i),layer)
                        if i == p_layer:
                                break
                self.contentFunc = model     
                del vgg16, cnn, checkpoint

        def getloss(self, fakeIm, realIm):
                if isinstance(fakeIm, numpy.ndarray):
                        fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
                        realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
                
                f_fake = self.contentFunc.forward(fakeIm)
                f_real = self.contentFunc.forward(realIm)
                f_real_no_grad = f_real.detach()
                loss = self.criterion(f_fake, f_real_no_grad)
                return loss

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer
def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################
class MyConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm=None, Res=False, sn=False, Leaky=False):
        super(MyConv2d, self).__init__()
        model = [nn.ReflectionPad2d(padding)]

        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]

        if norm == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        elif norm == 'Batch':
            model += [nn.BatchNorm2d(n_out)]
        elif norm == 'Layer':
            model += [LayerNorm(n_out)]
        elif norm != 'None':
            raise NotImplementedError('not implemeted norm type')

        if Leaky:
            model += [nn.LeakyReLU(inplace=False)]
        else:
            model += [nn.ReLU(inplace=False)]

        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.Res = Res

    def forward(self, x):
        if self.Res:
            return self.model(x) + x
        else:
            return self.model(x)

######################################################################
## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return
    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


####################################################################
#--------------------- Spectral Normalization ---------------------
#    This part of code is copied from pytorch master branch (0.5.0)
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