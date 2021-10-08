import os
import yaml
import time
import torch
import torchvision
import numpy as np
from PIL import Image
from numpy import savetxt
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from torchvision.transforms import ToPILImage, Compose

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Saver():
    def __init__(self, args, test=False):
        self.args = args
        if test:
            default_dir = os.path.join(args.result_dir, args.name)
        else:
            default_dir = os.path.join(args.experiment_dir, args.name)
        self.display_dir = os.path.join(default_dir, 'training_progress')
        self.model_dir = os.path.join(default_dir, 'models')
        self.image_dir = os.path.join(default_dir, 'down_results')
        self.kernel_dir = os.path.join(default_dir, 'estimated_kernels')

        if args.edsr_format:
            self.image_dir = os.path.join(default_dir, args.name)

        self.img_save_freq = args.img_save_freq
        self.model_save_freq = args.model_save_freq

        ## make directory
        if not os.path.exists(self.display_dir): os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir): os.makedirs(self.image_dir)
        if not os.path.exists(self.kernel_dir): os.makedirs(self.kernel_dir)
        
        if args.joint:
            self.image_sr_dir = os.path.join(default_dir, 'sr_results')
            if not os.path.exists(self.image_sr_dir): os.makedirs(self.image_sr_dir)

        config = os.path.join(default_dir,'config.yml')
        with open(config, 'w') as outfile:
            yaml.dump(args.__dict__, outfile, default_flow_style=False)

    ## save result images
    def write_img_down(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.png' % (self.display_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.png' % (self.display_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    ## save result images
    def write_img_LR(self, ep, num, model, args, fn):
        result_savepath = os.path.join(self.image_dir, 'ep_%03d'%ep)
        filename = fn[0].split('.')[0]
    
        if args.edsr_format:
            if args.scale == '2':
                scale = 'x2'
            elif args.scale == '4':
                scale = 'x4'
            else:
                raise NotImplementedError('Scale 2 and 4 are only available.')
            result_savepath = os.path.join(self.image_dir, scale)
        else:
            scale = ''

        if not os.path.exists(result_savepath):
            os.mkdir(result_savepath)

        images_list = model.get_outputs()
        
        img_filename = os.path.join(result_savepath, '%s%s.png'%(filename, scale))
        torchvision.utils.save_image(images_list[1] / 2 + 0.5, img_filename, nrow=1)

    ## save result images
    def write_img_SR(self, ep, sr, filename):
        result_savepath = os.path.join(self.image_sr_dir, 'ep_%03d'%ep)

        if not os.path.exists(result_savepath):
            os.mkdir(result_savepath)

        img_filename = os.path.join(result_savepath, filename[0])

        torchvision.utils.save_image(sr, img_filename, nrow=1)
        
    ## save model
    def write_model_down(self, ep, total_it, model):
        if ep != -1:
            print('save the down model @ ep %d' % (ep))
            model.state_save('%s/training_down_%04d.pth' % (self.model_dir, ep), ep, total_it)
            model.model_save('%s/model_down_%04d.pth' % (self.model_dir, ep), ep, total_it)
        else:
            model.state_save('%s/training_down_last.pth' % (self.model_dir), ep, total_it)
            model.model_save('%s/model_down_last.pth' % (self.model_dir), ep, total_it)

    def write_model_sr(self, ep, total_it, model):
        if ep != -1:
            print('save the sr model @ ep %d' % (ep))
            model.state_save('%s/training_sr_%04d.pth' % (self.model_dir, ep), ep, total_it)
            model.model_save('%s/model_sr_%04d.pth' % (self.model_dir, ep), ep, total_it)
        else:
            model.state_save('%s/training_sr_last.pth' % (self.model_dir), ep, total_it)
            model.model_save('%s/model_sr_last.pth' % (self.model_dir), ep, total_it)

    ## visualzie estimated kernel
    def write_kernel(self, ep, kernel):
        kernel_np = np.array(kernel.cpu())
        savetxt(os.path.join(self.kernel_dir, 'kernel_%02d.csv'%(ep)), kernel_np, delimiter=',')

        kernel /= kernel.abs().max()
        k_pos = kernel * (kernel > 0).float()
        k_neg = kernel * (kernel < 0).float()
        k_rgb = torch.stack([-k_neg, k_pos, torch.zeros_like(k_pos)], dim=0)
        pil = TF.to_pil_image(k_rgb.cpu())
        pil = pil.resize((self.args.adl_ksize * 20, self.args.adl_ksize * 20), resample=Image.NEAREST)
        pil.save(os.path.join(self.kernel_dir, 'kernel_%02d.png'%(ep))) # save kernel png

