# This file implements PAN plus (2R) model for MRI reconstruction

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
import time
from helpers_PAN import *
import quant
import quant_1bit

#%%

parser = ArgumentParser(description='QPAN')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=11, help='number of layers in QPAN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--bits', type=str, default='1', help='quantization bits from {1, 2, 3}, fp for full precision model')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {1, 4, 10}')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--batch_size', type=int, default='4', help='batch size, if GPU memory is low, choose 1')

args = parser.parse_args()

layer_num = args.layer_num
cs_ratio = args.cs_ratio
start_epoch = args.start_epoch
end_epoch =  args.end_epoch
learning_rate = args.learning_rate
model_dir = args.model_dir

test_name = 'BrainImages_test'

bits = args.bits
if not bits == 'fp':
    bits = int(bits)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

training_size = 800
batch_size = args.batch_size

quant_epoch = 9 # start quantizing the bits after this epoch
print('CS ratio is: ', cs_ratio)

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']

mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)

Training_data_Name = 'Training_BrainImages_256x256_100.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']

#%%     Architecture of the PAN plus 2R model

class Layer_plus(torch.nn.Module):
    def __init__(self):
        super(Layer_plus, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.lambda_ft = nn.Parameter(torch.Tensor([0.5]))
        self.gamma = nn.Parameter(torch.Tensor([1.1]))
        
        self.lambda_scad = nn.Parameter(torch.Tensor([0.5]))
        self.a_scad = nn.Parameter(torch.Tensor([2.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x_ft = firm_thresh(x_forward, self.lambda_ft, self.gamma)
        theta = 1/2
        x = theta*x_st + (1 - theta)*x_ft

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]

#   PAN plus 2R model 
class PAN_plus_2R(torch.nn.Module):
    def __init__(self, LayerNo):
        super(PAN_plus_2R, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(Layer_plus())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]


#%%

model = PAN_plus_2R(layer_num)
# model = nn.DataParallel(model)
model = model.to(device)
print_flag = 0   # print parameter number

if not bits == 'fp':
    target_weights = []
    for para in model.parameters():
        if len(para.size()) > 1:
            target_weights.append(para)
    
    bits_list = [bits] * len(target_weights)
    if bits == 1:
        Q = quant_1bit.quant(target_weights, b = bits_list)
    else:
        Q = quant.quant(target_weights, b = bits_list)
    
    print('The quantization to '+str(bits)+' has began')
else:
    print('Full precision training has began')


train_loader = DataLoader(dataset=RandomDataset(Training_labels , training_size), batch_size=batch_size, num_workers=0, shuffle=True)

model_dir = 'model'
data_dir = 'data'
log_dir = 'log'

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/MRI_PAN_layer_%d_ratio_%d" % (model_dir, layer_num,  cs_ratio)

print(model_dir)


if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for i in range(layer_num):
    print(model.fcs[i].lambda_step)


print('number of epochs is', end_epoch)

import time
start_time = time.time()
tot_loss_all_list = []
valid_loss_list = []
isQuantized = False

# Training loop
for epoch_i in range(start_epoch, end_epoch):
    
    print('Epoch:', epoch_i)
    
    tot_loss_all = 0
    tot_loss_discrepancy = 0
    tot_loss_constraint = 0
    
    for data in train_loader:

        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])

        PhiTb = FFT_Mask_ForBack()(batch_x, mask)

        [x_output, loss_layers_sym] = model(PhiTb, mask)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
        
        with torch.no_grad():
            tot_loss_all += loss_all.data
            tot_loss_discrepancy += loss_discrepancy.data
            tot_loss_constraint += torch.mul(gamma, loss_constraint)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        
        if isQuantized and not bits == 'fp':
            Q.restoreW()
        
        # plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        if epoch_i > quant_epoch and not bits == 'fp':
            # apply the quantization
            Q.apply()
            isQuantized = True
     
            
    tot_loss_all_list.append(tot_loss_all)
    print(tot_loss_all)
    
    if epoch_i % 3 == 0:
        output_data = "[%02d/%02d] Total Loss: %.10f, Discrepancy Loss: %.10f,  Constraint Loss: %.10f\n" % (epoch_i, end_epoch, tot_loss_all, tot_loss_discrepancy, tot_loss_constraint)
        print(output_data)
        
        # save the models
        torch.save(model.state_dict(), "./%s/net_params_bits_%s_2R_epoch_%d.pkl" % (model_dir, str(bits), epoch_i))  # save only the parameters


time_elapsed = time.time() - start_time
print('Time took for ', end_epoch, '- epochs ;', training_size ,'-training size; is' , time_elapsed ,'seconds' )

