import os
import math
import time
import torch
import model
import utility
import numpy as np
import torch.nn as nn
import torch.nn.utils as utils

from model import common

from tqdm import tqdm
from PIL import Image
from decimal import Decimal
from torch.optim.adam import Adam
from torchvision.models.vgg import vgg19
from model.Discriminator import Discriminator


class SRModel(nn.Module):
    def __init__(self, args, train=True):
        super(SRModel, self).__init__()

        self.args = args
        self.scale = args.scale
        self.gpu = 'cuda'
        self.error_last = 1e8

        self.training_type = args.training_type
        print('sr model : ',self.args.sr_model)
        print('training type : ',self.training_type)

        # define model, optimizer, scheduler, loss
        self.gen = model.Model(args)
        
        if args.pretrain_sr is not None:
            checkpoint = torch.load(args.pretrain_sr, map_location=lambda storage, loc: storage)
            self.gen.load_state_dict(checkpoint)
            print('Load pretrained SR model from {}'.format(args.pretrain_sr))

        if train:
            self.gen_opt = Adam(self.gen.parameters(), lr=args.lr_sr, betas=(0.9, 0.999))
            self.gen_sch = torch.optim.lr_scheduler.StepLR(self.gen_opt, args.decay_batch_size_sr, gamma=0.5) #args.gamma)
        
            self.content_criterion = nn.L1Loss().to(self.gpu)

            if self.training_type == 'esrgan':
                self.dis = Discriminator(args).to(self.gpu)
                self.dis_opt = Adam(self.dis.parameters(), lr=args.lr_sr, betas=(0.9, 0.999))
                self.dis_sch = torch.optim.lr_scheduler.StepLR(self.dis_opt, args.decay_batch_size_sr, gamma=0.5)

                self.adversarial_criterion = nn.BCEWithLogitsLoss().to(self.gpu)
                self.perception_criterion = PerceptualLoss().to(self.gpu)
                self.dis.train()

            self.gen.train()

        self.gen_loss = 0
        self.recon_loss = 0

    ## update images to the model
    def update_img(self, lr, hr=None):
        self.img_lr = lr
        self.img_hr = hr

        self.gen_loss = 0
        self.recon_loss = 0

    def generate_HR(self):
        #self.img_lr *= 255
        self.img_gen = self.gen(self.img_lr, 0)
        #self.img_gen /= 255
        
    def update_G(self):
        # EDSR style
        if self.training_type == 'edsr':
            self.gen_opt.zero_grad()

            self.recon_loss = self.content_criterion(self.img_gen, self.img_hr) * 255.0 # compensate range of 0 to 1
            
            self.recon_loss.backward()
            self.gen_opt.step()            
            self.gen_loss = self.recon_loss.item()
            self.gen_sch.step()
        
        # ESRGAN style
        elif self.training_type == 'esrgan':
            if self.args.cycle_recon:
                raise NotImplementedError('Do not support using cycle reconstruction loss in ESRGAN training')
            real_labels = torch.ones((self.img_hr.size(0), 1)).to(self.gpu)
            fake_labels = torch.zeros((self.img_hr.size(0), 1)).to(self.gpu)

            # training generator
            self.gen_opt.zero_grad()

            score_real = self.dis(self.img_hr)
            score_fake = self.dis(self.img_gen)
            
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()

            adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, fake_labels)
            adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, real_labels)
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            perceptual_loss = self.perception_criterion(self.img_gen, self.img_hr)
            content_loss = self.content_criterion(self.img_gen, self.img_hr) * 255.0 # compensate range of 0 to 1

            gen_loss = adversarial_loss * self.args.adv_w + \
                        perceptual_loss * self.args.per_w + \
                        content_loss * self.args.con_w

            gen_loss.backward()
            self.gen_loss = gen_loss.item()
            self.gen_opt.step()

            # training discriminator
            self.dis_opt.zero_grad()

            score_real = self.dis(self.img_hr)
            score_fake = self.dis(self.img_gen.detach())
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()

            adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, real_labels)
            adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, fake_labels)
            discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            discriminator_loss.backward()
            self.dis_opt.step()

            self.gen_sch.step()
            self.dis_sch.step()
        else:
            raise NotImplementedError('training type is not possible')

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(checkpoint['gen'])
        if train:
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            if self.training_type == 'esrgan':
                self.dis.load_state_dict(checkpoint['dis'])
                self.dis_opt.load_state_dict(checkpoint['dis_opt'])
        return checkpoint['ep'], checkpoint['total_it']                

    def state_save(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict(),
                 'gen_opt': self.gen_opt.state_dict(),
                 'ep': ep,
                 'total_it': total_it
                }
        if self.training_type == 'esrgan':
            state['dis'] = self.dis.state_dict(),
            state['dis_opt'] = self.dis_opt.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return

    def model_save(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict()}
        if self.training_type == 'esrgan':
            state['dis'] = self.dis.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss
