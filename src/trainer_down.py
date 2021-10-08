import time
import math
import torch
import networks
from PIL import Image
import torch.nn as nn
from filters import find_kernel
from data_loss import get_data_loss
from utility import get_gaussian_kernel, get_avgpool_kernel

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AdaptiveDownsamplingModel(nn.Module):
    def __init__(self, args):
        super(AdaptiveDownsamplingModel, self).__init__()

        self.args = args
        self.gpu = args.gpu
        self.data_loss_type = args.data_loss_type # data loss option

        self.gen = networks.G_Module(args, norm=args.gen_norm, nl_layer=networks.get_non_linearity(layer_type='lrelu'))    # generator
        self.gen.apply(networks.gaussian_weights_init)
        self.gen.cuda(args.gpu)

        self.down_filter = None

        if self.args.phase == 'train':
            self.dis = networks.D_Module(args, norm=args.dis_norm) # discriminators        
            self.dis.apply(networks.gaussian_weights_init) 
            self.dis.cuda(args.gpu)

            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=args.lr_down, betas=(0.9, 0.999))
            self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=args.lr_down, betas=(0.9, 0.999))

            self.gen_sch = torch.optim.lr_scheduler.StepLR(self.gen_opt, args.decay_batch_size_down, gamma=0.5)
            self.dis_sch = torch.optim.lr_scheduler.StepLR(self.dis_opt, args.decay_batch_size_down, gamma=0.5)
        
            self.ratio = args.ratio
            print('Data loss type : ', args.data_loss_type)

    ## update images to model
    def update_img(self, img_s, img_t=None):
        self.img_s = img_s.cuda(self.args.gpu).detach()
        if img_t is not None:
            self.img_t = img_t.cuda(self.args.gpu).detach()

        self.loss_dis = 0
        self.loss_gen = 0
        self.loss_data = 0

    ## generating LR iamges
    def generate_LR(self, scale='2'):
        if scale == '2':
            self.img_gen = self.gen(self.img_s)
        elif scale == '4':
            self.img_gen = self.gen(self.img_gen)
        else:
            raise NotImplementedError('scale is only available for [2, 4]')

    ## update discriminator D
    def update_D(self):
        self.dis_opt.zero_grad()

        loss_D = self.backward_D_gan(self.dis, self.img_t, self.img_gen)

        self.loss_dis = loss_D.item()

        self.dis_opt.step()
        self.dis_sch.step()
             
    ## update generator G
    def update_G(self, SRM_recon_loss=0):
        self.gen_opt.zero_grad()

        loss_gan = self.backward_G_gan(self.img_gen, self.dis)
        loss_data = get_data_loss(self.img_s, self.img_gen, self.data_loss_type, self.down_filter, self.args) * self.ratio

        if self.args.cycle_recon:
            loss_G = loss_gan + loss_data + SRM_recon_loss * self.args.cycle_recon_ratio
        else:
            loss_G = loss_gan + loss_data

        loss_G.backward() #retain_graph=True)

        self.loss_gen = loss_gan.item()
        self.loss_data = loss_data.item()

        self.gen_opt.step()
        self.gen_sch.step()

    ## loss function for discriminator D
    ## real to ones, and fake to zeros
    def backward_D_gan(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a).clamp(min=0.0, max=1.0)
            out_real = torch.sigmoid(out_b).clamp(min=0.0, max=1.0)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu).clamp(min=0.0, max=1.0)
            all1 = torch.ones_like(out_real).cuda(self.gpu).clamp(min=0.0, max=1.0)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def backward_G_gan(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G

    ## estimated 2d kernel with linear approximation
    def find_kernel(self):
        return find_kernel(self.img_s[0], self.img_gen[0], scale=2, k=self.args.adl_ksize, max_patches=-1)

    ## averaging estimated kernel for stabilization
    def stack_kernel(self, cnt, kernel):
        if cnt == 1:
            self.estimated_kernel = kernel
        else:
            self.estimated_kernel += kernel

        return self.estimated_kernel / float(cnt)

    ## ADL; update data loss with retrieved kernel
    ## customize 2d convolution filter weight with estimated 2d kernel,
    ## and set require_grad as False.
    def update_dataloss(self):
        channels = 3
        kernel_size = self.args.adl_ksize
        my_kernel = self.estimated_kernel

        my_kernel = my_kernel / torch.sum(my_kernel) # sum to one

        my_kernel = my_kernel.view(1, 1, kernel_size, kernel_size)
        my_kernel = my_kernel.repeat(channels, 1, 1, 1)

        my_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

        my_filter.weight.data = my_kernel
        my_filter.weight.requires_grad = False

        self.down_filter = my_filter
    

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(checkpoint['gen'])
        if train:
            self.dis.load_state_dict(checkpoint['dis'])
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])

            if self.data_loss_type == 'adl' and (checkpoint['ep']+1 > self.args.adl_interval):
                channels = 3
                kernel_size = self.args.adl_ksize
                self.down_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False)
                self.down_filter.load_state_dict(checkpoint['down_filter'])
                self.down_filter.weight.requires_grad = False
                self.down_filter = self.down_filter.cuda()

        return checkpoint['ep'], checkpoint['total_it']

    def state_save(self, filename, ep, total_it):
        state = {'dis': self.dis.state_dict(),
                 'gen': self.gen.state_dict(),
                 'dis_opt': self.dis_opt.state_dict(),
                 'gen_opt': self.gen_opt.state_dict(),
                 'ep': ep,
                 'total_it': total_it
                }
        if self.data_loss_type == 'adl' and (self.down_filter is not None):
            state['down_filter'] = self.down_filter.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return

    def model_save(self, filename, ep, total_it):
        state = {'dis': self.dis.state_dict(),
                 'gen': self.gen.state_dict(),
                }
        time.sleep(5)
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_source = self.img_s.detach()

        images_target = torch.zeros_like(self.img_s) # template
        margin = (self.img_s.shape[2] - self.img_t.shape[2]) // 2
        images_target[:, :, margin:-margin, margin:-margin] = self.img_t.detach()

        images_generated = torch.zeros_like(self.img_s) # template
        margin = (self.img_s.shape[2] - self.img_gen.shape[2]) // 2
        images_generated[:, :, margin:-margin, margin:-margin] = self.img_gen.detach()

        images_blank = torch.zeros_like(self.img_s).detach() # blank

        row1 = torch.cat((images_source[0:1, ::], images_blank[0:1, ::], images_generated[0:1, ::]),3)
        row2 = torch.cat((images_target[0:1, ::], images_blank[0:1, ::], images_blank[0:1, ::]),3)

        return torch.cat((row1,row2),2)

    def get_outputs(self):
        img_s = self.img_s.detach()
        img_gen = self.img_gen.detach()

        return [img_s, img_gen] 
