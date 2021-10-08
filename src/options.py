import argparse
from datetime import datetime


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ## dataset related
        # learning downsampling
        self.parser.add_argument('--source', type=str, default='Source', help='Source type')
        self.parser.add_argument('--target', type=str, default='Target', help='target type')
        # validation set for SR, only used in joint training
        self.parser.add_argument('--test_hr', type=str, help='HR images for validating')
        self.parser.add_argument('--test_lr', type=str, help='LR images for validating')
        ## data loader related
        self.parser.add_argument('--train_dataroot', type=str, default='../datasets/', help='path of train data')
        self.parser.add_argument('--test_dataroot', type=str, default='../datasets/', help='path of test data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=24, help='batch size')
        self.parser.add_argument('--patch_size_down', type=int, default=128, help='cropped image size for learning downsampling')
        self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')
        self.parser.add_argument('--flip', action='store_true', help='specified if flip')
        self.parser.add_argument('--rot', action='store_true', help='specified if rotate')
        self.parser.add_argument('--nobin', action='store_true', help='specified if not use bin')
        ## ouptput related
        self.parser.add_argument('--name', type=str, default='', help='folder name to save outputs')
        self.parser.add_argument('--experiment_dir', type=str, default='../experiments', help='path for saving result images and models')
        self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
        self.parser.add_argument('--make_down', action='store_true', help='specified if test')

        ## training related
        # common
        self.parser.add_argument('--gpu', type=str, default='cuda', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--scale', type=str, choices=('2', '4'), default='2', help='scale to SR, only support [2, 4]')
        # learning downsampling
        self.parser.add_argument('--resume_down', type=str, default=None, help='load training states for resume the downsampling learning')
        self.parser.add_argument('--epochs_down', type=int, default=80, help='number of epochs for training downsampling')
        self.parser.add_argument('--lr_down', type=float, default=0.00005, help='learning rate for learning downsampling')
        #self.parser.add_argument('--lr_policy', type=str, default='step', help='type of learn rate decay')
        self.parser.add_argument('--decay_batch_size_down', type=int, default=400000, help='decay batch size for learning downsampling') # currently, not using
        self.parser.add_argument('--dis_norm', type=str, default='Instance', help='normalization layer in discriminator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--gen_norm', type=str, default='Instance', help='normalization layer in generator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--cycle_recon', action='store_true', help='use self reconstruction loss for training downsampler, not that only available with jointly sr training case')
        self.parser.add_argument('--cycle_recon_ratio', type=float, default=0.1, help='hyper parameter for self reconstruction loss')
        # training SR
        self.parser.add_argument('--joint', action='store_true', help='jointly training downsampler and SR network')
        self.parser.add_argument('--pretrain_sr', type=str, default=None, help='load pretrained SR model for stable SR learning')
        self.parser.add_argument('--resume_sr', type=str, default=None, help='load training states for resume the downsampling learning')
        self.parser.add_argument('--epochs_sr_start', type=int, default=41, help='start epochs for training SR')
        self.parser.add_argument('--epochs_sr_end', type=int, default=80, help='end epochs for training SR')
        self.parser.add_argument('--lr_sr', type=float, default=0.00010, help='learning rate for training SR')
        self.parser.add_argument('--adv_w', type=float, default=0.01, help='weight for adversarial loss in esrgan training')
        self.parser.add_argument('--per_w', type=float, default=1.0, help='weight for adversarial loss in esrgan training')
        self.parser.add_argument('--con_w', type=float, default=0.1, help='weight for adversarial loss in esrgan training')
        self.parser.add_argument('--noise', action='store_true', help='inject noise in SR training')
        self.parser.add_argument('--noise_std', type=float, default=5.0, help='injected std of noise')
        self.parser.add_argument('--decay_batch_size_sr', type=int, default=50000, help='decay batch size for training SR')
        self.parser.add_argument('--sr_model', type=str, choices=('edsr','rrdb'), default='edsr', help='choose model to SR')
        self.parser.add_argument('--training_type', type=str, default='edsr', choices=('edsr', 'esrgan'), help='choose training type of SR')
        self.parser.add_argument('--precision', type=str, choices=('single','half'), default='singe', help='precision for forwarding SR')
        self.parser.add_argument('--realsr', action='store_true', help='just make SR image without calculating PSNR')
        self.parser.add_argument('--baseline', action='store_true', help='just train SR network with bicubic downsampled image')
        self.parser.add_argument('--patch_size_sr', type=int, default=128, help='cropped image size for learning sr')
        self.parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')
        self.parser.add_argument('--test_range', type=str, default='1-10', help='test data range')
        
        ## experimnet related
        self.parser.add_argument('--save_snapshot', type=int, default = 20, help='save snapshot')
        self.parser.add_argument('--save_log', action='store_true', help='enable saving log option')
        self.parser.add_argument('--save_results', action='store_true', help='enable saving intermediate image option')
        self.parser.add_argument('--save_intermodel', action='store_true', help='enable saving intermediate model option')
        self.parser.add_argument('--edsr_format', type=bool, default=False, help='save image as EDSR format')

        ## data loss related
        self.parser.add_argument('--data_loss_type', type=str, choices=('adl', 'lfl', 'bic', 'gau'), default='adl', help='type of available data type')
        # lfl
        self.parser.add_argument('--box_size', type=int, default=16, help='box size for filtering')
        # adl
        self.parser.add_argument('--adl_interval', type=int, default=10, help='update interval of data loss')
        self.parser.add_argument('--adl_ksize', type=int, default=20, help='kernel size for kernel estimation')
        self.parser.add_argument('--num_for_kernel_estimate', type=int, default=50, help='number of image to estimate kernel')
        # gau
        self.parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='gaussian std')
        self.parser.add_argument('--gaussian_ksize', type=int, default=16, help='gaussian kernel size')
        self.parser.add_argument('--gaussian_dense', action='store_true', help='option for dense gaussian')
        # balance b/w adv loss
        self.parser.add_argument('--ratio', type=float, default=100, help='ratio between adv loss and data loss')


    def parse(self):
        self.opt = self.parser.parse_args()

        if self.opt.name == '':
            self.opt.name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+self.opt.phase+'_'+self.opt.data_loss_type+'_'+self.opt.source+'_'+self.opt.target

        return self.opt

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ## dataset related
        # learning downsampling
        self.parser.add_argument('--source', type=str, default='Source', help='Source type')
        self.parser.add_argument('--target', type=str, default='Target', help='target type')
        # validation set for SR, only used in joint training
        self.parser.add_argument('--test_hr', type=str, help='HR images for validating')
        self.parser.add_argument('--test_lr', type=str, help='LR images for validating')
        ## data loader related
        self.parser.add_argument('--train_dataroot', type=str, default='../datasets/', help='path of train data')
        self.parser.add_argument('--test_dataroot', type=str, default='../datasets/', help='path of test data')
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')
        self.parser.add_argument('--nobin', action='store_true', help='specified if not use bin')
        self.parser.add_argument('--flip', action='store_true', help='specified if flip')
        self.parser.add_argument('--rot', action='store_true', help='specified if rotate')
        ## ouptput related
        self.parser.add_argument('--name', type=str, default='', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
        self.parser.add_argument('--make_down', action='store_true', help='specified if test')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')

        ## testing related
        # common
        self.parser.add_argument('--gpu', type=str, default='cuda', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--scale', type=str, choices=('2', '4'), default='2', help='scale to SR, only support [2, 4]')
        # testing downsampler
        self.parser.add_argument('--resume_down', type=str, default=None, help='load training states for resume the downsampling learning')
        self.parser.add_argument('--dis_norm', type=str, default='Instance', help='normalization layer in discriminator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--gen_norm', type=str, default='Instance', help='normalization layer in generator [None, Batch, Instance, Layer]')
        # testing SR
        self.parser.add_argument('--joint', type=bool, default=True, help='always set true in test mode')
        self.parser.add_argument('--pretrain_sr', type=str, default=None, help='load pretrained SR model for stable SR learning')
        self.parser.add_argument('--resume_sr', type=str, default=None, help='load training states for resume the downsampling learning')
        self.parser.add_argument('--sr_model', type=str, choices=('edsr','rrdb'), default='edsr', help='choose model to SR')
        self.parser.add_argument('--training_type', type=str, default='edsr', choices=('edsr', 'esrgan'), help='choose training type of SR')
        self.parser.add_argument('--precision', type=str, choices=('single','half'), default='singe', help='precision for forwarding SR')
        self.parser.add_argument('--realsr', action='store_true', help='just make SR image without calculating PSNR')
        self.parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')
        self.parser.add_argument('--test_range', type=str, default='1-10', help='test data range')
        ## experimnet related
        self.parser.add_argument('--save_log', action='store_true', help='enable saving log option')
        self.parser.add_argument('--save_results', action='store_true', help='enable saving intermediate image option')
        self.parser.add_argument('--edsr_format', type=bool, default=False, help='save image as EDSR format')

        ## data loss related
        self.parser.add_argument('--data_loss_type', type=str, choices=('adl', 'lfl', 'bic', 'gau'), default='adl', help='type of available data type')
        # lfl
        self.parser.add_argument('--box_size', type=int, default=16, help='box size for filtering')
        # adl
        self.parser.add_argument('--adl_ksize', type=int, default=20, help='kernel size for kernel estimation')
        self.parser.add_argument('--num_for_kernel_estimate', type=int, default=50, help='number of image to estimate kernel')
        # gau
        self.parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='gaussian std')
        self.parser.add_argument('--gaussian_ksize', type=int, default=16, help='gaussian kernel size')
        self.parser.add_argument('--gaussian_dense', action='store_true', help='option for dense gaussian')


    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- loading options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        
        if self.opt.name == '':
            self.opt.name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+self.opt.phase

        return self.opt
