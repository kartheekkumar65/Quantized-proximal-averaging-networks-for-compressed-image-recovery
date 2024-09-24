# This file implements PAN (2R) model

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
import quant
import quant_1bit

#%%  Initializing the hyper parameters

parser = ArgumentParser(description='QPAN')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=101, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='number of layers in QPAN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--bits', type=str, default='1', help='quantization bits from {1, 2, 3}, fp for full precision model')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10}')
parser.add_argument('--model_type', type=str, default='3R', help='1R for Q-ISTA-Net, 2R for Q-PAN 2R, 3R for Q-PAN 3R')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--result_dir', type=str, default='data', help='results directory')
parser.add_argument('--test_name', type=str, default='BSD68', help='name of test set')

args = parser.parse_args()

layer_num = args.layer_num
cs_ratio = args.cs_ratio
test_name = args.test_name
start_epoch = args.start_epoch
end_epoch =  args.end_epoch
learning_rate = args.learning_rate
model_dir = args.model_dir

bits = args.bits
if not bits == 'fp':
    bits = int(bits)

model_type = args.model_type

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 64
training_size = 88912

quant_epoch = 9 # start quantizing the bits after this epoch

print('CS ratio is: ', cs_ratio)

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio) #(args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']

Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (args.matrix_dir, cs_ratio)

X_data = Training_labels.transpose()
Y_data = np.dot(Phi_input, X_data)
Y_YT = np.dot(Y_data, Y_data.transpose())
X_YT = np.dot(X_data, Y_data.transpose())
Qinit_new = np.dot(X_YT, np.linalg.inv(Y_YT))



if os.path.exists(Qinit_Name):
    print('if')
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']

else:
    print('else')
    X_data = Training_labels.transpose()
    Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})
    

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)


#%%     Architecture of the PAN 2R model


# Structure of alyer in PAN 2R model
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


    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

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

        x_pred = x_pred.view(-1, 1089)

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

        for i in range(LayerNo):
            onelayer.append(Layer_plus())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]


#%% Start the training

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


# Define the training dataloader
train_loader = DataLoader(dataset=RandomDataset(Training_labels , training_size), batch_size=batch_size, num_workers=0, shuffle=True)

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define the place where trained models are to be saved
model_dir = 'model'
data_dir = 'data'
log_dir = 'log'

model_dir = "./%s/CS_PAN_plus_layer_%d_ratio_%d" % (model_dir, layer_num, cs_ratio)

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
    
    tot_loss_all = 0
    tot_loss_discrepancy = 0
    tot_loss_constraint = 0
    
    for data in train_loader:

        batch_x = data
        batch_x = batch_x.to(device)

        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(device)

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
    
    if epoch_i % 10 == 0:
        output_data = "[%02d/%02d] Total Loss: %.10f, Discrepancy Loss: %.10f,  Constraint Loss: %.10f\n" % (epoch_i, end_epoch, tot_loss_all, tot_loss_discrepancy, tot_loss_constraint)
        print(output_data)
        
        # save the models
        torch.save(model.state_dict(), "./%s/net_params_bits_%s_2R_epoch_%d.pkl" % (model_dir, str(bits), epoch_i))  # save only the parameters

time_elapsed = time.time() - start_time
print('Time took for ', end_epoch, '- epochs ;', training_size ,'-training size; is' , time_elapsed ,'seconds' )















