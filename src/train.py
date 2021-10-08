import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from saver import Saver
from options import Options
from trainer_sr import SRModel
from bicubic_pytorch import core
from trainer_down import AdaptiveDownsamplingModel
from dataset import unpaired_dataset, paired_dataset
from utility import log_writer, plot_loss_down, plot_psnr, timer, calc_psnr, quantize, _normalize

## parse options
parser = Options()
args = parser.parse()

## data loader
print('preparing dataset ...')
dataset = unpaired_dataset(args, phase='train')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nThreads)
dataset = unpaired_dataset(args, phase='test')
test_loader_down = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)
if args.joint and (args.test_lr is not None):
    dataset = paired_dataset(args)
    test_loader_sr = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)
ep0 = 0
total_it = 0

## SR model only if joint training
if args.joint:
    print('\nMaking SR-Model... ')
    SRM = SRModel(args)

## Adaptive Downsampling Model
print('\nMaking Adpative-Downsampling-Model...')
ADM = AdaptiveDownsamplingModel(args)
if args.resume_down is not None:
    ep0, total_it = ADM.resume(args.resume_down)
    print('\nLoad downsampling model from {}'.format(args.resume_down))
if args.resume_sr is not None:
    ep0, total_it = SRM.resume(args.resume_sr)
    print('\nLoad SR model from {}'.format(args.resume_sr))
 
## saver and training log
saver = Saver(args)
data_timer, train_timer_down, kernel_estimator_timer = timer(), timer(), timer()
if args.joint:
    train_timer_sr, eval_timer_sr = timer(), timer()
training_log = log_writer(args.experiment_dir, args.name)

## losses
loss_dis = []  # discriminator loss
loss_gen = []  # generator loss
loss_data = [] # data loss
if args.joint:
    psnrs = [] # L1 loss for SR

max_epochs = max(args.epochs_down, args.epochs_sr_end) if args.joint else args.epochs_down

print('\ntraining start')
for ep in range(ep0, max_epochs):
    sr_txt1, sr_txt2 = '', ''
    if args.joint and ((ep+1) in range(args.epochs_sr_start, args.epochs_sr_end+1)):
        sr_txt1 = ' SR lr %.06f' % (SRM.gen_opt.param_groups[0]['lr'])
        sr_txt2 = '  SR loss  |'

    training_log.write('\n[ epoch %03d/%03d ]   G lr %.06f D lr %.06f%s'
            % (ep+1, max_epochs ,ADM.gen_opt.param_groups[0]['lr'], ADM.dis_opt.param_groups[0]['lr'], sr_txt1))
    print_txt = '|    Progress   |    Dis    |    Gen    |    data    |%s' % (sr_txt2)
    training_log.write('-'*len(print_txt))
    training_log.write(print_txt)

    loss_dis_item = 0
    loss_gen_item = 0
    loss_data_item = 0
    if args.joint:
        loss_sr_item = 0
    cnt = 0

    data_timer.tic()
    for it, (img_s, img_t, _) in enumerate(train_loader): 
        if img_t.size(0) != args.batch_size:
            continue
        data_timer.hold()

        train_timer_down.tic()
        ADM.update_img(img_s, img_t)
        ADM.generate_LR()                           
        train_timer_down.hold()

        ## train downsampling network ADM
        train_timer_down.tic()
        if ((ep+1) in range(0,args.epochs_down+1)) and (not args.baseline):
            ADM.update_D()                           
            if args.cycle_recon:
                img_lr, img_hr = ADM.img_gen.clamp(min=-1.0, max=1.0), ADM.img_s.detach()    
                img_lr, img_hr = _normalize(img_lr, img_hr) # normalize [-1,1] to [0,1]

                img_lr = quantize(img_lr, fake=True)

                SRM.update_img(img_lr, img_hr)    
                SRM.generate_HR()
                SRM.calculate_grad()
                ADM.update_G(SRM_recon_loss=SRM.recon_loss)
            else:
                ADM.update_G()
        train_timer_down.hold()

        ## if joint training is enabled, train SR network SRM with generated image by ADM
        sr_loss_txt, sr_timer_txt = '', ''
        if args.joint and ((ep+1) in range(args.epochs_sr_start, args.epochs_sr_end+1)):
            train_timer_sr.tic()
            if args.baseline: # use bicubic
                img_lr, img_hr = core.imresize(ADM.img_s, scale=0.5).detach(), ADM.img_s.detach()
            else:
                if args.scale == '4':
                    [ _x ] = _normalize(ADM.img_gen) # normalize [-1,1] to [0,1]
                    _x = quantize(_x)
                    [ _x ] = _normalize(_x, mul=2, add=-0.5, reverse=True) # normalize [0,1] to [-1,1]
                    ADM.img_gen = _x
                    ADM.generate_LR(scale=args.scale)                           
                img_lr, img_hr = ADM.img_gen.detach().clamp(min=-1.0, max=1.0), ADM.img_s.detach()
            img_lr, img_hr = _normalize(img_lr, img_hr) # normalize [-1,1] to [0,1]

            if args.noise:
                n = args.noise_std * torch.Tensor(np.random.normal(size=img_lr.shape)).cuda() / 255.0
                img_lr = (img_lr + n).clamp(max=1.0, min=0.0)

            #b = max(args.patch_size_down - args.patch_size_sr, 0) // 4
            #if ( b != 0 ): img_lr, img_hr = img_lr[:,:,b:-b, b:-b], img_hr[:,:,2*b:-2*b, 2*b:-2*b]
            
            img_lr = quantize(img_lr)

            SRM.update_img(img_lr, img_hr)    
            SRM.generate_HR()
            SRM.update_G()               

            loss_sr_item += SRM.gen_loss
            train_timer_sr.hold()

        loss_dis_item += ADM.loss_dis
        loss_gen_item += ADM.loss_gen
        loss_data_item += ADM.loss_data
        cnt += 1
    
        ## print training log with save
        if (it+1) % (len(train_loader)//10) == 0:
            if args.joint and ((ep+1) in range(args.epochs_sr_start, args.epochs_sr_end+1)):
                loss_sr_item_avg = loss_sr_item/cnt
                sr_loss_txt = '  %0.5f  |' % (loss_sr_item_avg)
                sr_timer_txt = '  +%.01fs' % (train_timer_sr.release())
                loss_sr_item = 0


            loss_dis_item_avg = loss_dis_item/cnt
            loss_gen_item_avg = loss_gen_item/cnt
            loss_data_item_avg = loss_data_item/cnt
            training_log.write('|   %04d/%04d   |  %.05f  |  %.05f  |  %.06f  |%s  %.01f+%.01fs %s'
                                % ( (it+1), len(train_loader), loss_dis_item_avg, loss_gen_item_avg,
                                    loss_data_item_avg, sr_loss_txt,
                                    train_timer_down.release(), data_timer.release(), sr_timer_txt))
            loss_dis_item = 0
            loss_gen_item = 0
            loss_data_item = 0
            cnt = 0

            if args.save_results:
                saver.write_img_down(ep*len(train_loader) + (it+1), ADM)

        data_timer.tic()
    training_log.write('-'*len(print_txt))

    loss_dis.append(loss_dis_item_avg)
    loss_gen.append(loss_gen_item_avg)
    loss_data.append(loss_data_item_avg)
    plot_loss_down(os.path.join(args.experiment_dir, args.name), loss_dis, loss_gen, loss_data)

    
    ## 2d linear kernel estimating
    kernel_estimator_timer.tic()
    ADM.eval()
    with torch.no_grad():
        for cnt, (img_s, _, _) in enumerate(test_loader_down):
            img_s = img_s[:, :, 0:min(img_s.shape[2], 1000), 0:min(img_s.shape[3], 1000)] # for memory efficiency
            ADM.update_img(img_s)
            ADM.generate_LR()
            kernel = ADM.find_kernel() # estimate 2d kernel of current generator network
            estimated_kernel = ADM.stack_kernel(cnt+1, kernel) # stack to average retrieved 2d kernel
            if cnt == args.num_for_kernel_estimate: # not use all of test set to save computational cost
                break
        saver.write_kernel(ep+1, estimated_kernel)
        if (args.data_loss_type == 'adl') and ((ep+1) != args.epochs_down) and ((ep+1) % args.adl_interval == 0):
            training_log.write('Data Loss Update with Estimated kernel at %02d'%(ep+1))
            ADM.update_dataloss()
    training_log.write('Total time to estimate kernel: %.01f s'%kernel_estimator_timer.toc())
    ADM.train()


    ## sr evaluation for joint training
    if args.joint and ((ep+1) in range(args.epochs_sr_start, args.epochs_sr_end+1)) and (args.test_lr is not None):
        if (not args.realsr) or args.save_results: 
            eval_timer_sr.tic()
            SRM.eval()
            psnr_sum = 0
            cnt = 0
            with torch.no_grad():
                for img_hr, img_lr, fn in tqdm(test_loader_sr, ncols=80):
                    img_hr, img_lr = img_hr.cuda(), img_lr.cuda()
                    SRM.update_img(img_lr)
                    SRM.generate_HR()
                    
                    img_sr = quantize(SRM.img_gen)
                    if args.save_results:
                        saver.write_img_SR(ep, img_sr, fn)

                    if not args.realsr:
                        psnr_sum += calc_psnr(
                                img_sr, img_hr, args.scale 
                            )
                        cnt += 1
            eval_timer_sr.hold()
            if not args.realsr:
                training_log.write('PSNR on test set: %.04f, %.01fs' % (psnr_sum/(cnt), eval_timer_sr.release()))
                psnrs.append(psnr_sum/(cnt))
                plot_psnr(os.path.join(args.experiment_dir, args.name), psnrs)
            else:
                training_log.write('Total time elapsed: %.01fs' % (eval_timer_sr.release()))
            SRM.train()

    if (ep+1) % args.save_snapshot == 0:
        saver.write_model_down(ep+1, total_it+1, ADM)
        if args.joint and ((ep+1) in range(args.epochs_sr_start, args.epochs_sr_end+1)):
            saver.write_model_sr(ep+1, total_it+1, SRM)

    ## Save last model and state
    training_log.write('Saving last model and training state..')
    saver.write_model_down(-1, total_it+1, ADM)
    if args.joint and ((ep+1) in range(args.epochs_sr_start, args.epochs_sr_end+1)):
        saver.write_model_sr(-1, total_it+1, SRM)  

        
if args.make_down:
    print('\nmaking downwampling images ...')
    ADM.eval()
    with torch.no_grad():
        for number, (img_s, _, fn) in enumerate(test_loader_down):
            ADM.update_img(img_s)
            ADM.generate_LR()
            if args.scale == '4':  
                ADM.generate_LR(scale=args.scale)
            saver.write_img_LR(ep+1, (number+1), ADM, args, fn)
    ADM.train()
    print('\ndone!')

## Save network weights
#saver.write_model(ep+1, total_it+1, ADM)

