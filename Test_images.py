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


from ISTANet_arch import ISTANet, ISTANet_plus
from PAN_2R_arch import PAN_2R, PAN_2R_plus
from PAN_2R_arch_fp import PAN_2R_fp, PAN_2R_plus_fp
from PAN_3R_arch import PAN_3R, PAN_3R_plus

# In[ ]:

parser = ArgumentParser(description='QPAN')

parser.add_argument('--start_epoch', type=int, default=60, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=101, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='number of layers in QPAN')
parser.add_argument('--bits', type=str, default='1', help='quantization bits from {1, 2, 3}, fp for full precision model')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10}')
parser.add_argument('--model_type', type=str, default='3R', help='1R for Q-ISTA-Net, 2R for Q-PAN 2R, 3R for Q-PAN 3R')
parser.add_argument('--isPlus', type=bool, default='True', help='plus models or non-plus models')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--result_dir', type=str, default='result', help='results directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()

layer_num = args.layer_num
cs_ratio = args.cs_ratio
test_name = args.test_name
start_epoch = args.start_epoch
end_epoch =  args.end_epoch
result_dir = args.result_dir

bits = args.bits
if not bits == 'fp':
    bits = int(bits)

if bits == 'fp':
    start_epoch = 100

model_type = args.model_type
isPlus = args.isPlus

if isPlus:
    plus_type = 'plus'
else:
    plus_type = 'reg'

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('CS ratio is: ', cs_ratio)

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']


Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (args.matrix_dir, cs_ratio)

if not isPlus:
    model_dir = "./%s/CS_PAN_layer_%d_ratio_%d" % (args.model_dir, layer_num, cs_ratio)
else:
    model_dir = "./%s/CS_PAN_plus_layer_%d_ratio_%d" % (args.model_dir, layer_num, cs_ratio)



# In[ ]:


if os.path.exists(Qinit_Name):
    print('if')
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']

else:
    Training_data_Name = 'Training_Data.mat'
    Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name)) 
    Training_labels = Training_data['labels']

    X_data = Training_labels.transpose()
    Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})


# In[ ]:


Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)



#%%  Testing phase

if not isPlus: 
    if model_type == '1R':
        model = ISTANet(layer_num)
        print('QISTA Net model ')
    elif model_type == '2R':
        if bits == 'fp':
            model = PAN_2R_fp(layer_num)
        else:
            model = PAN_2R(layer_num)
        print('QPAN 2R model ')
    elif model_type == '3R':
        model = PAN_3R(layer_num)
        print('QPAN 3R model ')
    else:
        raise('choose proper model')
else:
    if model_type == '1R':
        model = ISTANet_plus(layer_num)
        print('QISTA Net plus model ')
    elif model_type == '2R':
        if bits == 'fp':
            model = PAN_2R_plus_fp(layer_num)
        else:
            model = PAN_2R_plus(layer_num)
        print('QPAN 2R plus model ')
    elif model_type == '3R':
        model = PAN_3R_plus(layer_num)
        print('QPAN 3R plus model ')
    else:
        raise('choose proper model')



model = model.to(device)

test_dir = os.path.join(args.data_dir, test_name)

if test_name == 'Set11':
    filepaths = glob.glob(test_dir + '/*.tif')
elif test_name == 'BSD68':
    filepaths = glob.glob(test_dir + '/*.jpg')

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)



prefix = 'module.'
n_clip = len(prefix)
loaded_dict = torch.load("./%s/net_params_bits_%s_%s_epoch_%d.pkl" % (model_dir, str(bits), model_type, start_epoch), map_location=device)

adapted_dict = {}
for k, v in loaded_dict.items():
    if k.startswith(prefix):
        adapted_dict[k[n_clip:]] = v
    else:
        adapted_dict[k] = v

model.load_state_dict(adapted_dict)

print("\n CS Reconstruction Start")
print(torch.unique(model.fcs[0].conv1_forward.detach()))

for epoch_num in range(start_epoch, end_epoch , 10):
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
    
    with torch.no_grad():
        # print('#'*100)
        if not ImgNum > 0:
            raise('No images loaded')
        for img_no in range(ImgNum):
            # print(img_no)
    
            imgName = filepaths[img_no]
            Img = cv2.imread(imgName, 1)
    
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()
    
            Iorg_y = Img_yuv[:,:,0]
    
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Icol = img2col_py(Ipad, 33).transpose()/255.0
    
            Img_output = Icol
    
    
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
    
            Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
    
            [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
    
            Prediction_value = x_output.cpu().data.numpy()
    
            X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)
    
            rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)
            
            Img_rec_yuv[:,:,0] = X_rec*255
    
            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
    
            resultName = imgName.replace(args.data_dir, result_dir)[:-4]
            # resultName = imgName[:-4]
            
            file_name = "%s_QPAN_%s_bits_%s_%s_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.bmp" % (resultName,  model_type, str(bits), plus_type, cs_ratio, epoch_num, rec_PSNR, rec_SSIM)
            cv2.imwrite(file_name, im_rec_rgb)
            
    
            del x_output
    
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM

    
        output_data = "\n bits - %s Model Type %s CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (str(bits),model_type, cs_ratio, test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
        print(output_data)

