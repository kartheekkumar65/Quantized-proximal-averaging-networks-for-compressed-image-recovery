#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import time
from helpers_PAN import *
from helpers_PAN import *
import matplotlib.pyplot as plt


from ISTANet_arch import ISTANet_MRI, ISTANet_MRI_plus
from PAN_2R_arch import PAN_MRI_2R, PAN_MRI_2R_plus
from PAN_2R_arch_fp import PAN_MRI_2R_fp, PAN_MRI_2R_plus_fp
from PAN_3R_arch import PAN_MRI_3R, PAN_MRI_3R_plus

# In[ ]:

parser = ArgumentParser(description='QPAN')

parser.add_argument('--start_epoch', type=int, default=180, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=11, help='number of layers in QPAN')
parser.add_argument('--bits', type=str, default='fp', help='quantization bits from {1, 2, 3}, fp for full precision model')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {1, 4, 10}')
parser.add_argument('--model_type', type=str, default='3R', help='1R for Q-ISTA-Net, 2R for Q-PAN 2R, 3R for Q-PAN 3R')
parser.add_argument('--isPlus', type=bool, default='True', help='plus models or non-plus models')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--result_dir', type=str, default='result', help='results directory')

args = parser.parse_args()

layer_num = args.layer_num
cs_ratio = args.cs_ratio
start_epoch = args.start_epoch
end_epoch =  args.end_epoch

test_name = 'BrainImages_test'
device = 'cuda:0'

bits = args.bits
if not bits == 'fp':
    bits = int(bits)
    
if bits == 'fp':
    start_epoch = 198

model_type = args.model_type
isPlus = args.isPlus

if isPlus:
    plus_type = 'plus'
else:
    plus_type = 'reg'

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]

epoch_num = start_epoch

print('CS ratio is: ', cs_ratio)

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']


mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)

if not isPlus:
    model_dir = "./%s/MRI_PAN_layer_%d_ratio_%d" % (args.model_dir, layer_num, cs_ratio)
else:
    model_dir = "./%s/MRI_PAN_plus_layer_%d_ratio_%d" % (args.model_dir, layer_num, cs_ratio)


#%%  Testing phase

if not isPlus: 
    if model_type == '1R':
        model = ISTANet_MRI(layer_num)
        print('QISTA Net model ')
    elif model_type == '2R':
        if bits == 'fp':
            model = PAN_MRI_2R_fp(layer_num)
        else:
            model = PAN_MRI_2R(layer_num)
        print('QPAN MRI 2R model ')
    elif model_type == '3R':
        model = PAN_MRI_3R(layer_num)
        print('QPAN MRI 3R model ')
    else:
        raise('choose proper model')
else:
    if model_type == '1R':
        model = ISTANet_MRI_plus(layer_num)
        print('QISTA Net MRI plus model ')
    elif model_type == '2R':
        if bits == 'fp':
            model = PAN_MRI_2R_plus_fp(layer_num)
        else:
            model = PAN_MRI_2R_plus(layer_num)
        print('QPAN MRI 2R plus model ')
    elif model_type == '3R':
        model = PAN_MRI_3R_plus(layer_num)
        print('QPAN MRI 3R plus model ')
    else:
        raise('choose proper model')
    
    
model = model.to(device)

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

prefix = 'module.'
n_clip = len(prefix)
loaded_dict = torch.load("./%s/net_params_bits_%s_%s_epoch_%d.pkl" % (model_dir, str(bits), model_type, epoch_num), map_location=device)

adapted_dict = {}
for k, v in loaded_dict.items():
    if k.startswith(prefix):
        adapted_dict[k[n_clip:]] = v
    else:
        adapted_dict[k] = v

model.load_state_dict(adapted_dict)



print("\n CS Reconstruction Start")
#print(torch.unique(model.fcs[0].conv1_forward.detach()))

for epoch_num in range(start_epoch, end_epoch , 3):
    # print(epoch_num)
    prefix = 'module.'
    n_clip = len(prefix)
    loaded_dict = torch.load("./%s/net_params_bits_%s_%s_epoch_%d.pkl" % (model_dir, str(bits), model_type, epoch_num), map_location=device)
    
    adapted_dict = {}
    for k, v in loaded_dict.items():
        if k.startswith(prefix):
            adapted_dict[k[n_clip:]] = v
        else:
            adapted_dict[k] = v

    model.load_state_dict(adapted_dict)
    print("./%s/net_params_bits_%s_%s_epoch_%d.pkl" % (model_dir, str(bits), model_type, epoch_num))
    
    if not bits == 'fp': 
        if len(torch.unique(model.fcs[0].conv1_forward.detach())) > 2**bits and epoch_num > 10:
            raise('Quantization error')
        
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    
    Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    
    print('\n')
    print('-'*75)
    print("MRI CS Reconstruction Start")
    print(epoch_num)
    
    with torch.no_grad():
        for img_no in range(ImgNum):
        #for img_no in range(1,2):
    
            imgName = filepaths[img_no]
    
            Iorg = cv2.imread(imgName, 0)
    
            Icol = Iorg.reshape(1, 1, 256, 256) / 255.0
    
            Img_output = Icol
    
            start = time()
    
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
    
            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
    
            [x_output, loss_layers_sym] = model(PhiTb, mask)
    
            end = time()
    
            initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)
    
            Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)
    
            X_init = np.clip(initial_result, 0, 1).astype(np.float64)
            X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
    
            init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
            init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)
    
            rec_PSNR = psnr(X_rec*255., Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec*255., Iorg.astype(np.float64), data_range=255)
            
            im_rec_rgb = np.clip(X_rec*255, 0, 255).astype(np.uint8)
            
            resultName = imgName.replace(args.data_dir, args.result_dir)[:-4]
            # resultName = imgName[:-4]
            
            file_name = "%s_ACorNet_%s_bits_%s_%s_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.bmp" % (resultName,  model_type, str(bits), plus_type, cs_ratio, epoch_num, rec_PSNR, rec_SSIM)
            cv2.imwrite(file_name, im_rec_rgb)
                
            del x_output
    
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
    
            Init_PSNR_All[0, img_no] = init_PSNR
            Init_SSIM_All[0, img_no] = init_SSIM
    
    print('\n')
    output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
    print(output_data)

    print("MRI CS Reconstruction End")
    
#%% 

