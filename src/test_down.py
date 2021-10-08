import os
import torch
from saver import Saver
from options import TestOptions
from dataset import unpaired_dataset
from utility import quantize, _normalize
from trainer_down import AdaptiveDownsamplingModel


# parse options
parser = TestOptions()
args = parser.parse()

# test mode
args.batch_size = 1

# daita loader
print('\nmaking dataset ...')
dataset = unpaired_dataset(args, phase='test')
test_loader_down = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)

# model
print('\nmaking model ...')
ADM = AdaptiveDownsamplingModel(args)

if args.resume_down is None:
    raise NotImplementedError('put trained downsampling model for testing')
else:
    ep0, total_it = ADM.resume(args.resume_down, train=False)
ep0 += 1
print('load model successfully!')

saver = Saver(args, test=True)

print('\ntest start ...')
ADM.eval()
with torch.no_grad():
    for number, (img_s, _, fn) in enumerate(test_loader_down):
        
        if (number+1) % (len(test_loader_down)//10) == 0:
            print('[{:05d} / {:05d}] ...'.format(number+1, len(test_loader_down)))

        ADM.update_img(img_s)
        ADM.generate_LR()
        
        if args.scale == '4':
            [ _x ] = _normalize(ADM.img_gen) # normalize [-1,1] to [0,1]
            _x = quantize(_x)
            [ _x ] = _normalize(_x, mul=2, add=-0.5, reverse=True) # normalize [0,1] to [-1,1]
            ADM.img_gen = _x
            ADM.generate_LR(scale=args.scale)

        saver.write_img_LR(1, (number+1), ADM, args, fn)
print('\ntest done!')