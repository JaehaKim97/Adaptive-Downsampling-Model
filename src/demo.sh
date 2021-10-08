# retreiving downsampling kernel between HR image dataset 'Source' and LR image dataset 'Target'.
# generate downsampled version of 'Source' image from retrieved downsampling kernel saved in './experiments/save_name/down_results/'.
CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --make_down

### (Optional) joint trainig with SR model(default: edsr)
## edsr style training
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --joint

## esrgan style training
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --joint --training_type esrgan

## scale x4 training with RRDB model
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --joint --sr_model rrdb --scale 4

## specify sr training duration
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --joint --epochs_sr_start 41 --epochs_sr_end 80

## you can test your sr model on validation set if you have.
## note that validation set 'filename_lr' and 'filename_hr' should be located at './dataset/'.
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --joint --test_lr filename_lr --test_hr filename_hr

## you may only have target LR images, as common case for real-world target.
## in that case, you can visualize sr_results and check them in './experiments/save_name/sr_results/'.
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --name save_name --joint --test_lr filename_lr --realsr --save_results


## more demo scripts used for main paper are in below

## x2 scale sr for synthetic DIV2K target
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr filname_lr --test_hr filename_hr --save_results --sr_model edsr --scale 2 --epochs_sr_start 41 --training_type edsr --batch_size 24 --name synthetic_div2k-edsrx2

## x4 scale sr for synthetic DIV2K target
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr filename_lr --test_hr filename_hr --save_results --sr_model edsr --scale 4 --epochs_sr_start 41 --training_type edsr --batch_size 24 --name synthetic_div2k-edsrx4
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr filename_lr --test_hr filename_hr --save_results --sr_model rrdb --scale 4 --epochs_sr_start 61 --training_type edsr --batch_size 24 --pretrain_sr model/pretrain/rrdb_x4-9d40f7f7.pth  --name synthetic_div2k-rrdbx4

## x2 scale sr for realsr dataset
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr filename_lr --test_hr filename_hr --save_results --sr_model edsr --scale 2 --epochs_sr_start 41 --training_type edsr --batch_size 24 --name realsr-edsrx2 --noise

## x4 scale sr for realsr dataset
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr filename_lr --test_hr filename_hr --save_results --sr_model edsr --scale 4 --epochs_sr_start 41 --training_type edsr --batch_size 24 --noise --name realsr-edsrx4 --make_down
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr filename_lr --test_hr filename_hr --save_results --sr_model rrdb --scale 4 --epochs_sr_start 61 --training_type edsr --batch_size 24 --pretrain_sr model/pretrain/rrdb_x4-9d40f7f7.pth  --noise --name realsr-rrdbx4 --make_down

## x2 scale sr for dped dataset
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr Target --realsr --save_results --sr_model edsr --scale 2 --epochs_sr_start 41 --training_type edsr --batch_size 24 --name dped-edsrx2-edsr
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr Target --realsr --save_results --sr_model edsr --scale 2 --epochs_sr_start 41 --training_type esrgan --batch_size 24 --name dped-edsrx2-esrgan

## x4 scale sr for dped dataset
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr Target --realsr --save_results --sr_model edsr --scale 4 --epochs_sr_start 61 --training_type edsr --batch_size 24 --chop --name dped-edsrx4-edsr
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr Target --realsr --save_results --sr_model edsr --scale 4 --epochs_sr_start 61 --training_type esrgan --batch_size 24 --chop --name dped-edsrx4-esrgan
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr Target --realsr --save_results --sr_model rrdb --scale 4 --epochs_sr_start 61 --training_type edsr --batch_size 8 --chop --pretrain_sr model/pretrain/rrdb_x4-9d40f7f7.pth --name dped-rrdbx4-edsr
# CUDA_VISIBLE_DEVICES=0 python train.py --source Source --target Target --joint --test_lr Target --realsr --save_results --sr_model rrdb --scale 4 --epochs_sr_start 61 --training_type esrgan --batch_size 8 --chop --pretrain_sr model/pretrain/rrdb_x4-9d40f7f7.pth --name dped-rrdbx4-esrgan #--con_w 0.01