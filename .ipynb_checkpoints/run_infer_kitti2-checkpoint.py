# coding=gbk
import argparse
import os
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
#from scipy.ndimage import imread
from imageio import imread, imsave
from skimage.transform import resize as imresize
#from scipy.misc import imsave, imresize
from loss import *
import time
import random

import sys
sys.path.append('./third_party/cython')
from connectivity import enforce_connectivity

'''
Infer from bsds500 dataset:
author:Fengting Yang 
last modification:  Mar.14th 2019

usage:
1. set the ckpt path (--pretrained) and output
2. comment the output if do not need

results will be saved at the args.output
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='/root/autodl-tmp/mamba_s/data_preprocessing/KITTI/test',help='path to images folder')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model', default= '/root/autodl-tmp/mamba_s/CKPT/VMamba_stage2_lelu_double_ffm/BSD500/LEncoder1_adam_3000000epochs_epochSize6000_b16_lr0.0005_posW0.003_24_11_21_10_58/model_best.tar')
parser.add_argument('--output', metavar='DIR', default= './output/VMamba_stage2_kitti_double_ffm_best_png' ,help='path to output folder')

parser.add_argument('--downsize', default=16, type=float, help='superpixel grid cell, must be same as training setting')
parser.add_argument('-b', '--batch-size', default=1, type=int,  metavar='N', help='mini-batch size')

# the BSDS500 has two types of image, horizontal and veritical one, here I use train_img and input_img to presents them respectively
parser.add_argument('--train_img_height', '-t_imgH', default=320,  type=int, help='img height must be 16*n')
parser.add_argument('--train_img_width', '-t_imgW', default=480,  type=int, help='img width must be 16*n')
parser.add_argument('--input_img_height', '-v_imgH', default=480,  type=int, help='img height_must be 16*n')
parser.add_argument('--input_img_width', '-v_imgW', default=1280,   type=int, help='img width must be 16*n')

args = parser.parse_args()
args.test_list = args.data_dir + '/test.txt'


random.seed(100)
@torch.no_grad()
def test(args, model, img_paths, save_path, spixeIds, idx, scale):
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    img_ = imread(load_path)
    H_, W_, _ = img_.shape
    img = cv2.resize(img_, (int(args.input_img_width), int(args.input_img_height)), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)

    # The orignal sz of nyu test set 448*608
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(int(375), int(1242)), mode='nearest').type(
        torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
                                                    n_spixels=2400 * scale * scale, b_enforce_connect=True)
    n_spixel = len(np.unique(spixel_label_map))
    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))

    # save spixel viz

    # if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
    #     os.makedirs(os.path.join(save_path, 'spixel_viz'))
    # spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    # imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))
    
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    spixel_viz = np.clip(spixel_viz, 0, 1)  # Clip values to [0, 1] if normalized
    spixel_viz = (spixel_viz * 255).astype(np.uint8)  # Convert to uint8 for saving
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

    # save the unique maps as csv
    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))
    output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
    # plus 1 to make it consistent with the toolkit format
    np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i', delimiter=",")

    if idx % 10 == 0:
        print("processing %d" % idx)

    return toc, n_spixel


def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))
    mean_time_list = []
    input_img_height = args.input_img_height
    input_img_width = args.input_img_width
    # The spixel number we test
    for scale in [0.6, 0.7]:#[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        assert (320 * scale % 16 == 0 and 480 * scale % 16 == 0)
        save_path = args.output + '/test_multiscale_enforce_connect/SPixelNet_nSpixel_{0}'.format(int(input_img_height/16 * scale * input_img_width /16 * scale))

        args.input_img_height, args.input_img_width = input_img_height * scale, input_img_width * scale

        print('=> will save everything to {}'.format(save_path))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        tst_lst = []
        with open(args.test_list, 'r') as tf:
            img_path = tf.readlines()
            for path in img_path:
                tst_lst.append(path[:-1])

        print('{} samples found'.format(len(tst_lst)))

        # create model
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained model '{}'".format(network_data['arch']))
        model = models.__dict__[network_data['arch']](data=network_data).cuda()
        model.eval()
        args.arch = network_data['arch']
        cudnn.benchmark = True

        spixlId, XY_feat_stack = init_spixel_grid(args, b_train=False)

        mean_time = 0
        for n in range(len(tst_lst)):
            time,n_spixel = test(args, model, tst_lst, save_path, spixlId, n, scale)
            mean_time += time
        mean_time /= len(tst_lst)
        mean_time_list.append((n_spixel, mean_time))
        print("for spixel number {}: with mean_time {} , generate {} spixels".format(int(20 * scale * 30 * scale), mean_time, n_spixel))

    with open(args.output + '/test_multiscale_enforce_connect/mean_time.txt', 'w+') as f:
        for item in mean_time_list:
            tmp = "{}: {}\n".format(item[0], item[1])
            f.write(tmp)


if __name__ == '__main__':
        main()
