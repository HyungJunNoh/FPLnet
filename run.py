# import libraries
import argparse
#from doctest import testfile

import os
#from turtle import st
import numpy as np
import math
#from pyparsing import lineStart

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utilities.model import UNet
from utilities.dataset import *
from utilities.util import *

import matplotlib.pyplot as plt
#from matplotlib.ticker import ScalarFormatter

from torchvision import transforms, datasets

import time
from tqdm import tqdm

import smtplib
from email.mime.text import MIMEText

torch.set_default_dtype(torch.float64)
# Parser
parser = argparse.ArgumentParser(description="Train the UNet", 
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument("--name", default='0727_64', type=str, dest="名前")

parser.add_argument("--lr", default=0.0005, type=float, dest="lr")
parser.add_argument("--batch_size", default=199, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=5000, type=int, dest="num_epoch")

#parser.add_argument("--data_dir", default="./FPL_116", type=str, dest="data_dir")

#parser.add_argument("--mode", default="test", type=str, dest="mode")
#parser.add_argument("--train_continue", default="on", type=str, dest="train_continue")
#parser.add_argument("--gpu_parallel", default="off", type=str, dest="gpu_parallel")
parser.add_argument("--gpu_num", default="0", type=int, dest="gpu_num")
parser.add_argument("--early_stopping", default="5000", type=int, dest="early_stopping")

args = parser.parse_args()

# set training parameters
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

#名前 = args.名前
#data_dir = args.data_dir

#mode = args.mode
#train_continue = args.train_continue
#gpu_parallel = args.gpu_parallel
GPU_NUM = args.gpu_num
patience = args.early_stopping

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
'''
lst_data = os.listdir(data_dir)
ckpt_list = [f for f in lst_data if f.startswith('checkpoint_%s'%名前)]
#log_list = [f for f in lst_data if f.startswith('log_%s'%名前)]
result_list = [f for f in lst_data if f.startswith('result_%s'%名前)]
ckpt_list.sort(); log_list.sort(); result_list.sort()
ckpt_dir = os.path.join(data_dir, ckpt_list[-1])
#log_dir = os.path.join(data_dir, log_list[-1])
result_dir = os.path.join(data_dir, result_list[-1])
'''

ckpt_dir = './utilities/model_epoch1150.pth'
result_dir = './result'
"""
# Make dir
if not os.path.exists(result_dir):
	os.makedirs(os.path.join(result_dir, 'png'))
	os.makedirs(os.path.join(result_dir, 'numpy'))
"""

transform = transforms.Compose([Normalization(min=0.00, max=770.0), ToTensor()])
dataset_test = Dataset(data_dir='./dataset', transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)

# Generate network
net = UNet().to(device)

# Loss functions
fn_loss = nn.MSELoss().to(device)

def fn_psnr(output, label):
	batch_psnr = []
	label = label[:,0,:,:]
	output = output[:,0,:,:]
	for i in range(label.shape[0]):
		label_squeezed = label[i,:,:]
		output_squeezed = output[i,:,:]

		max = torch.max(output_squeezed)
		min = torch.min(output_squeezed)
		s = max - min
		MSE_loss = fn_loss(output_squeezed, label_squeezed)

		psnr = 20*math.log10(torch.abs(s))-10*math.log10(MSE_loss)
		batch_psnr += [psnr]

	PSNR = np.mean(batch_psnr)
	return PSNR

# Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)

# Scheduler (for lr deacy)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)

# other functions
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)	
fn_totorch = lambda x: torch.Tensor(x).to(device)
fn_denorm = lambda x, min, max: (x * (max-min)) + min
fn_norm = lambda x, min, max: (x-min)/(max-min)

# Training
st_epoch = 0
MSE_best = 10000.0

# Initializing early_stopping object
early_stopping = EarlyStopping(patience = patience, verbose = False)

# Train mode
if MSE_best == -4259545677:
	print("No training mode allowed")

# Test mode
else:
	net, optim, st_epoch, MSE_best = load(ckpt_dir=ckpt_dir, net=net, optim=optim, MSE_best=MSE_best)

	with torch.no_grad():
		net.eval()
		loss_arr = []
		avg_psnr = []
		#Conservative_loss needs label_filename, label, output, label_ori, input_for_subtract
		save_input_for_subtract = []
		save_label_ori = []
		save_output = []
		save_label = []
		
		energy_error = []
		density_error = []
		momentum_error = []
		target_energy_error = []
		target_density_error = []
		target_momentum_error = []
		temperature_z = []
		temperature_r = []
		target_temperature_z = []
		target_temperature_r = []

		#Time series figure
		Time_energy = []
		Time_momentum = []
		Time_density = []
		Time_TZ = []
		Time_TR = []
		Time_TZ_target = []
		Time_TR_target = []
		nanamae = 'hello'
		Time_energy2 = []
		Time_momentum2 = []
		Time_density2 = []
		Time_TZ2 = []
		Time_TR2 = []
		Time_TZ_target2 = []
		Time_TR_target2 = []
		nanamae2 = 'hello'
		for batch, data in tqdm(enumerate(loader_test, 1), desc='Test'):
			# forward pass
			Tlabel_ori = data['label'].to(device)
			Tinput_for_train = data['input'].to(device)
			label_filename = data['label_filename']
			for i in range(Tlabel_ori.shape[0]):

				name = label_filename[i]
				fac = name[6:14]
				step = name[17:20]
				
				if step == '001':
					step = int(step)
					input_for_train = Tinput_for_train[i].clone().detach()
					input_for_train = input_for_train[np.newaxis,:,:,:]

					save_input_for_subtract.clear()
					save_label_ori.clear()
					save_output.clear()
					save_label.clear()

					while step < 200:
						# update answers
						label_ori = Tlabel_ori[step-1]
						label_ori = label_ori[np.newaxis,:,:,:]
						input_ans = Tinput_for_train[step-1]
						input_ans = input_ans[np.newaxis,:,:,:]

						input_dist = input_ans[:,0,:,:]
						input_dist = input_dist[:,np.newaxis,:,:]

						input_for_subtract = fn_denorm(input_dist, min=0.00, max=770.0)
						label = label_ori - input_for_subtract

						output = net(input_for_train)

						# calculate loss function
						loss = fn_loss(output, label)

						PSNR = fn_psnr(output, label)

						avg_psnr.append(PSNR)

						# save for full timestep diagnostics
						save_input_for_subtract.append(input_for_subtract)
						save_label_ori.append(label_ori)
						save_output.append(output)
						save_label.append(label)

						Ploss = WT_Conservative_loss(1e+18, 1, 1, label_filename, label, output, label_ori, input_for_subtract)
						E_loss, E_error, Target_E_loss, D_loss, Target_D_loss, M_loss, Target_M_loss, TZ, TR, TZ_target, TR_target = Ploss['E_loss'], Ploss['E_error'], Ploss['Target_E_loss'], Ploss['D_loss'], Ploss['Target_D_loss'], Ploss['M_loss'], Ploss['Target_M_loss'], Ploss['TZ'], Ploss['TR'], Ploss['TZ_target'], Ploss['TR_target']

						energy_error.append(E_error.item())
						density_error.append(D_loss.item())
						momentum_error.append(M_loss.item())
						target_energy_error.append(Target_E_loss.item())
						target_density_error.append(Target_D_loss.item())
						target_momentum_error.append(Target_M_loss.item())
						temperature_z.append(TZ[0].item())
						temperature_r.append(TR[0].item())
						target_temperature_z.append(TZ_target[0].item())
						target_temperature_r.append(TR_target[0].item())
      
						# Save the kekka
						temp_ic = name[6:14]
						sq_label = label.squeeze().detach().cpu().numpy()
						sq_input = input_for_subtract.squeeze().detach().cpu().numpy()
						sq_output = output.squeeze().detach().cpu().numpy()
      
						np.save(os.path.join(result_dir, 'numpy', 'label_%s%03d.npy' % (temp_ic,step)), sq_label)
						np.save(os.path.join(result_dir, 'numpy', 'input_%s%03d.npy' % (temp_ic,step)), sq_input)
						np.save(os.path.join(result_dir, 'numpy', 'output_%s%03d.npy' % (temp_ic,step)), sq_output)
						
						denorm_input = fn_denorm(input_for_train[0,0,:,:], min=0.00, max=770.0)
						input_for_train[0,0,:,:] = denorm_input + output[0,0,:,:]
						input_for_train = fn_norm(input_for_train, min=0.00, max=770.0)

						step += 1
						# Time series loss
						if label_filename[0] == 'label_0.795000_00001.npy':
							Time_energy.append(E_error.item())
							Time_density.append(D_loss.item())
							Time_momentum.append(M_loss.item())
							nanamae = label_filename[0]
							Time_TZ.append(temperature_z[step])
							Time_TR.append(temperature_r[step])
							Time_TZ_target.append(target_temperature_z[step])
							Time_TR_target.append(target_temperature_r[step])

						if label_filename[0] == 'label_1.980000_00001.npy':
							Time_energy2.append(E_error.item())
							Time_density2.append(D_loss.item())
							Time_momentum2.append(M_loss.item())
							nanamae2 = label_filename[0]
							Time_TZ2.append(temperature_z[step])
							Time_TR2.append(temperature_r[step])
							Time_TZ_target2.append(target_temperature_z[step])
							Time_TR_target2.append(target_temperature_r[step])

					# Dignostic of 1st and the last simulation
					print("\nlen(save_label): ", len(save_label))
					#print(label_filename)
					print('\n')
					label = save_label[-1]
					output = save_output[-1]
					label_ori = save_label_ori[-1]
					input_for_subtract = save_input_for_subtract[0]

		print('Total data point: ', len(energy_error))
		print(max(density_error), max(momentum_error), max(energy_error))
		print(len(avg_psnr), ' Avg PSNR: ', np.mean(avg_psnr))
		avg_psnr = []
		# time series plot		
		t = range(1,200)
		plt.plot(t, Time_energy, color='#6515E8', linestyle='--', label='$T_{\parallel}/T_{\perp}=0.795, E$')
		plt.plot(t, Time_density, color='#FFBE3F', linestyle='--', label='$T_{\parallel}/T_{\perp}=0.795, n$')
		plt.plot(t, Time_momentum, color='#FF0074', linestyle='--', label='$T_{\parallel}/T_{\perp}=0.795, P$')
		plt.plot(t, Time_energy2, color='#6515E8', linestyle='solid', label='$T_{\parallel}/T_{\perp}=1.98, E$')
		plt.plot(t, Time_density2, color='#FFBE3F', linestyle='solid', label='$T_{\parallel}/T_{\perp}=1.98, n$')
		plt.plot(t, Time_momentum2, color='#FF0074', linestyle='solid', label='$T_{\parallel}/T_{\perp}=1.98, P$')
		plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
		plt.axis([0,200,0,0.0006])
		plt.legend()
		plt.xlabel('Time Step')
		plt.ylabel('Error')
		plt.show()

		# Temperature plot
		t = range(1,198)
		plt.plot(t, Time_TZ_target[:-2], color='black', linestyle='solid', label='$T_{\parallel}^{Ground Truth}$')
		plt.plot(t, Time_TR_target[:-2], color='black', linestyle='solid', label='$T_{\perp}^{Ground Truth}$')
		plt.plot(t, Time_TZ[:-2], color='red', linestyle='--', label='$T_{\parallel}$')
		plt.plot(t, Time_TR[:-2], color='blue', linestyle='--', label='$T_{\perp}$')
		plt.xlabel('Time Step')
		plt.ylabel('Temperature [eV]')
		plt.legend()
		plt.show()
		plt.xlabel('Time Step')
		plt.ylabel('Temperature [eV]')
		Differ_one = np.array(Time_TZ_target[:-2])-np.array(Time_TZ[:-2])
		Differ_two = np.array(Time_TR_target[:-2])-np.array(Time_TR[:-2])
		print(np.max(np.abs(Differ_one)))
		print(np.max(np.abs(Differ_two)))
		t = range(1,200)
		plt.plot(t, Time_TZ2, color='#6515E8', linestyle='--', label='1.98$T_{\parallel}$')
		plt.plot(t, Time_TR2, color='#FFBE3F', linestyle='--', label='1.98$T_{\perp}$')
		plt.plot(t, Time_TZ_target2, color='#6515E8', linestyle='solid', label='1.98$T_{\parallel}^{Ground Truth}$')
		plt.plot(t, Time_TR_target2, color='#FFBE3F', linestyle='solid', label='1.98$T_{\perp}^{Ground Truth}$')
		plt.legend()
		plt.show()

		bins2 = np.logspace(-10,-2,num=100)
		plt.hist(density_error, bins2, rwidth=0.8, color='red', alpha=0.5)
		plt.xscale("log")
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)
		plt.title("Mean : %s Median : %s"%( format(np.mean(np.abs(density_error)), '.4E'), format(np.median(np.abs(density_error)), '4E')))
		plt.axvline(np.mean(np.abs(density_error)), color='red', linestyle='dashed', linewidth=1)
		plt.axvline(np.median(np.abs(density_error)), color='blue', linestyle='dashed', linewidth=1)
		plt.show(); plt.clf()

		bins2 = np.logspace(-10,-2,num=100)
		plt.hist(momentum_error, bins2, rwidth=0.8, color='red', alpha=0.5)
		plt.xscale("log")
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)
		plt.title("Mean : %s Median : %s"%( format(np.mean(np.abs(momentum_error)), '.4E'), format(np.median(np.abs(momentum_error)), '4E')))
		plt.axvline(np.mean(np.abs(momentum_error)), color='red', linestyle='dashed', linewidth=1)
		plt.axvline(np.median(np.abs(momentum_error)), color='blue', linestyle='dashed', linewidth=1)
		plt.show(); plt.clf()

		bins2 = np.logspace(-10,-2,num=100)
		plt.hist(energy_error, bins2, rwidth=0.8, color='red', alpha=0.5)
		plt.xscale("log")
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)
		plt.title("Mean : %s Median : %s"%( format(np.mean(np.abs(energy_error)), '.4E'), format(np.median(np.abs(energy_error)), '4E')))
		plt.axvline(np.mean(np.abs(energy_error)), color='red', linestyle='dashed', linewidth=1)
		plt.axvline(np.median(np.abs(energy_error)), color='blue', linestyle='dashed', linewidth=1)
		plt.show(); plt.clf()

		# Target
		bins2 = np.logspace(-19,-11,num=100)
		plt.hist(target_density_error, bins2, rwidth=0.8, color='red', alpha=0.5)
		plt.xscale("log")
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)
		plt.title("Mean : %s Median : %s"%( format(np.mean(np.abs(target_density_error)), '.4E'), format(np.median(np.abs(target_density_error)), '4E')))
		plt.axvline(np.mean(np.abs(target_density_error)), color='red', linestyle='dashed', linewidth=1)
		plt.axvline(np.median(np.abs(target_density_error)), color='blue', linestyle='dashed', linewidth=1)
		plt.show(); plt.clf()

		bins2 = np.logspace(-19,-11,num=100)
		plt.hist(target_momentum_error, bins2, rwidth=0.8, color='red', alpha=0.5)
		plt.xscale("log")
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)
		plt.title("Mean : %s Median : %s"%( format(np.mean(np.abs(target_momentum_error)), '.4E'), format(np.median(np.abs(target_momentum_error)), '4E')))
		plt.axvline(np.mean(np.abs(target_momentum_error)), color='red', linestyle='dashed', linewidth=1)
		plt.axvline(np.median(np.abs(target_momentum_error)), color='blue', linestyle='dashed', linewidth=1)
		plt.show(); plt.clf()

		bins2 = np.logspace(-19,-11,num=100)
		plt.hist(target_energy_error, bins2, rwidth=0.8, color='red', alpha=0.5)
		plt.xscale("log")
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)
		plt.title("Mean : %s Median : %s"%( format(np.mean(np.abs(target_energy_error)), '.4E'), format(np.median(np.abs(target_energy_error)), '4E')))
		plt.axvline(np.mean(np.abs(target_energy_error)), color='red', linestyle='dashed', linewidth=1)
		plt.axvline(np.median(np.abs(target_energy_error)), color='blue', linestyle='dashed', linewidth=1)
		plt.show(); plt.clf()