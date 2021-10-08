import os
import torch
from tqdm import tqdm
from saver import Saver
from trainer_sr import SRModel
from options import TestOptions
from utility import timer, calc_psnr, quantize
from trainer_down import AdaptiveDownsamplingModel
from dataset import unpaired_dataset, paired_dataset

# parse options
parser = TestOptions()
args = parser.parse()

# test mode
saver = Saver(args, test=True)
args.batch_size = 1

# daita loader
dataset = paired_dataset(args)
test_loader_sr = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)

# model
SRM = SRModel(args, train=False)
if (args.resume_sr is None) and (args.pretrain_sr is None):
    raise NotImplementedError('put pretrained model for test')
elif args.resume_sr is not None:
    _, _ = SRM.resume(args.resume_sr, train=False)
print('load model successfully!')

eval_timer_sr = timer()

eval_timer_sr.tic()
SRM.eval()
ep0 = 0
psnr_sum = 0
cnt = 0
with torch.no_grad():
    for img_hr, img_lr, fn in tqdm(test_loader_sr, ncols=80):
        img_hr, img_lr = img_hr.cuda(), img_lr.cuda()
        if args.precision == 'half':
            img_lr = img_lr.half()

        SRM.update_img(img_lr)
        SRM.generate_HR()

        img_sr = quantize(SRM.img_gen)

        if args.save_results:
            saver.write_img_SR(ep0, img_sr, fn)

        if not args.realsr:
            psnr_sum += calc_psnr(
                    img_sr, img_hr, args.scale, rgb_range=1
                )
            cnt += 1

eval_timer_sr.hold()
if not args.realsr:
    print('PSNR on test set: %.04f, %.01fs' % (psnr_sum/(cnt), eval_timer_sr.release()))


