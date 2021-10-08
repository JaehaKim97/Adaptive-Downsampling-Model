# Adaptive-Downsampling-Model
This repository is an official PyTorch implementation of the paper **"Toward Real-World Super-Resolution via Adaptive Downsampling Models"** which is accepted at TPAMI([link](https://ieeexplore.ieee.org/document/9521710)).

## Dependencies
* Python 3.7
* PyTorch >= 1.6.0
* matplotlib
* imageio
* pyyaml
* scipy
* numpy
* tqdm
* PIL

In this project, we learn the adaptive downsampling model(**ADM**) with an unpaired dataset consisting of *HR* and *LR* images but not pixel aligned.

## 🚉: Dataset preperation

As *HR* is not responsible to pixel aligned with *LR*, we recommend to use [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset as *HR* dataset, which is consist of clean and high-resolution images.
You can download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).
We note that experiments conducted on our paper used 400 HR images('0001.png'-'0400.png') as *HR*.

For *LR* datasets, you should put a bunch of images that undergo a similar downsampling process. (e.g. [DPED](https://people.ee.ethz.ch/~ihnatova/), [RealSR](https://github.com/csjcai/RealSR), or you own images from same camera setting)

Please put *HR* and *LR* datasets in ```datasets/```. Again, different lengths between each dataset are acceptable, as we noted that two datasets are not responsible for pixel-aligned. However, we also note that total iterations of 1 epoch can differ along with your dataset size. Since our **ADM** learns average downsampling kernel along with *LR* datasets, please use available LR images with scene variety as much as possible for stable training(we recommend to use more than 200 for HD scale images). For more details on the effect of the number of training samples, please refer to our paper.

## 🚋: Training

### Learning to Downsample

Let denote filename of *HR* dataset as *Source*, and *LR* dataset as *Target*.

Our ADM will retrieve downsampling between two datasets *Source* and *Target*, and generate downsampled version of *Source* with retrieved downsampling kernel. Basic usage of training is following:

```
cd src
CUDA_VISIBLE_DEVICE=0 python train.py --source Source --target Target --name save_name --make_down
```

Generated downsampled version of *Source* will be saved at ```./experiments/save_name/down_results/```. Note that you can use *Source* and generated downsampled version of *Source* as **paired dataset** in conventional SR settings.

### Joint training with Image super-resolution (Optional)

Here we additionally support joint training with SR network, which use intermediate generated image as paired dataset hence does not require additional SR network training step. Usage of joint training is following:

```
cd src
CUDA_VISIBLE_DEVICE=0 python train.py --source Source --target Target --name save_name --joint
```

Default SR model is 'EDSR-baseline', but you can change with ```--sr_model```.

If you have validation datasets for evaluating perforamce of SR, please locate them in ```datasets/```. After then, you can measure performance of SR network by measuring PSNR on validation datasets. You can specify it with ```--test_lr filename_lr --test_hr filename_hr```, where *filename_hr* and *filename_lr* should be paired images.

In case that you don't have validation paired datasets as common in real-world, you can visualize SR results by ```--test_lr filename_lr --realsr --save_results```, and then SR results will be saved in ```./experiments/save_name/sr_results/```. Note that *filename_lr* can be same with *Target*.

You can check detailed usage of this repo in ```demo.sh``` .

Please note that experimental results reported in our paper are conducted in separate manner(i.e., we only generate downsampled images and train SR network with corresponding official implementation), so the results in joint training may slightly differs with number in the paper.

## BibTeX

    @ARTICLE{9521710,
        author={Son, Sanghyun and Kim, Jaeha and Lai, Wei-Sheng and Yang, Ming-Hsuan and Lee, Kyoung Mu},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
        title={Toward Real-World Super-Resolution via Adaptive Downsampling Models}, 
        year={2021},
        volume={},
        number={},
        pages={1-1},
        doi={10.1109/TPAMI.2021.3106790}
     }

## :e-mail: Contact

If you have any question, please email `jhkim97s2@gmail.com`.
