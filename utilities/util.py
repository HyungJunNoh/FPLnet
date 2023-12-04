# import libraries
import os
import numpy as np
import copy

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=100, verbose=False, trace_func=print):

		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.early_stop = False
		self.trace_func = trace_func
		self.val_loss_min = 100000
	def __call__(self, val_loss, net, ckpt_dir, optim, epoch, MSE_best):
		if val_loss > self.val_loss_min:
			self.counter += 1
			if self.verbose:
				self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.val_loss_min = val_loss
			self.save(ckpt_dir, net, optim, epoch, self.val_loss_min)
			self.counter = 0

	def save(self, ckpt_dir, net, optim, epoch, MSE_best):
		if self.verbose:
			self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {MSE_best:.6f}).  Saving model ...')
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)		
		torch.save({'net': net.state_dict(), 'optim': optim.state_dict(), 'MSE_best': MSE_best},
								"%s/model_epoch%d.pth" % (ckpt_dir, epoch))
		self.val_loss_min = MSE_best

# Load network
def load(ckpt_dir, net, optim, MSE_best):
	if not os.path.exists(ckpt_dir):
		epoch = 0
		MSE_best = 10000.0
		return net, optim, epoch, MSE_best

	ckpt_lst = os.listdir(ckpt_dir)
	ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

	net.load_state_dict(dict_model['net'])
	optim.load_state_dict(dict_model['optim'])
	epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
	MSE_best = dict_model['MSE_best']

	return net, optim, epoch, MSE_best

def diagnosis(diff_T_factor):

	device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

	mesh_Nr = 40
	mesh_Nz = 60   # Generally Nz = Nr*2 due to different domain length
	density = 10**19
	vol_real = 0.1
	echarge = 1.6022e-19  # Electron Charge
	mass = 1.6276e-27 # Kg
	eps_0 = 8.8542e-12
	Tr = 100 # Temperature unit : eV
	Tz = Tr * float(diff_T_factor)  # Note that Tr = 1/2(Tx+Ty), i.e. Equilibrium : Tx=Ty=Tz=Tr
	t_hat_step = 0.1   #Units : time_reference

	# System Parameter
	v_th_z = np.sqrt(Tz*echarge/mass)
	v_th_r = np.sqrt(2*Tr*echarge/mass)
	mesh_bd_factor = 1.05 # boundary = mesh_bd_factor * maximum velocity of particle ; 1.1

	bd_mesh_factor = 5 # unit: v_th
	vz_max = v_th_z * bd_mesh_factor
	vz_min = -vz_max
	vr_max = v_th_r * bd_mesh_factor

	mesh_zN = np.max([np.abs(vz_max), np.abs(vz_min)])*mesh_bd_factor
	mesh_z1 = -mesh_zN
	mesh_rN = vr_max*mesh_bd_factor
	mesh_Nrm1 = mesh_Nr-1
	mesh_r = np.linspace(0,mesh_rN,mesh_Nr)
	mesh_z = np.linspace(mesh_z1,mesh_zN,mesh_Nz)

	mesh_dr = mesh_r[1]-mesh_r[0]
	mesh_dz = mesh_z[1]-mesh_z[0]

	[mesh_Z, mesh_R]=np.meshgrid(mesh_z,mesh_r) #For mesh figure : 40x60 double

	dist_initial_ref = density*np.exp(-mesh_Z**2/(2*v_th_z**2) - mesh_R**2/(v_th_r**2)) / (np.sqrt(2*np.pi)*np.pi*(v_th_r**2)*v_th_z) #Bi-Maxwellian

	dist = dist_initial_ref

	numeric_T = mass*(((mesh_R**2 + mesh_Z**2)*dist*vol).sum(axis=0)).sum(axis=0) / (3*echarge*density)
	T_overall = numeric_T
	Coulomb_log= 23 - np.log(np.sqrt(2*density*1e-6/T_overall)/T_overall)
	col_factor = echarge**4 * Coulomb_log / ( 8 * np.pi * eps_0**2 * mass**2 )
	time_reference = 8 * np.pi * np.sqrt(2*mass) * eps_0*eps_0 * (T_overall*echarge)**1.5 / (density* echarge**4 * Coulomb_log)
	tstep = t_hat_step * time_reference

	fn_totorch = lambda x: torch.Tensor(x).to(device)

	vol = fn_totorch(vol)
	mesh_Z = fn_totorch(mesh_Z)
	mesh_R = fn_totorch(mesh_R)

	diag_params = {'tstep': tstep, 'vol': vol, 'mesh_Z': mesh_Z, 'mesh_R': mesh_R}

	return diag_params

def diagnosis_noTorch(diff_T_factor):

	mesh_Nr = 40
	mesh_Nz = 60   # Generally Nz = Nr*2 due to different domain length
	density = 10**19
	vol_real = 0.1
	echarge = 1.6022e-19  # Electron Charge
	mass = 1.6276e-27 # Kg
	eps_0 = 8.8542e-12
	Tr = 100 # Temperature unit : eV
	Tz = Tr * float(diff_T_factor)  # Note that Tr = 1/2(Tx+Ty), i.e. Equilibrium : Tx=Ty=Tz=Tr
	t_hat_step = 0.1   #Units : time_reference

	# System Parameter
	v_th_z = np.sqrt(Tz*echarge/mass)
	v_th_r = np.sqrt(2*Tr*echarge/mass)
	mesh_bd_factor = 1.05 # boundary = mesh_bd_factor * maximum velocity of particle ; 1.1

	bd_mesh_factor = 5 # unit: v_th
	vz_max = v_th_z * bd_mesh_factor
	vz_min = -vz_max
	vr_max = v_th_r * bd_mesh_factor

	mesh_zN = np.max([np.abs(vz_max), np.abs(vz_min)])*mesh_bd_factor
	mesh_z1 = -mesh_zN
	mesh_rN = vr_max*mesh_bd_factor
	mesh_Nrm1 = mesh_Nr-1
	mesh_r = np.linspace(0,mesh_rN,mesh_Nr)
	mesh_z = np.linspace(mesh_z1,mesh_zN,mesh_Nz)

	mesh_dr = mesh_r[1]-mesh_r[0]
	mesh_dz = mesh_z[1]-mesh_z[0]

	[mesh_Z, mesh_R]=np.meshgrid(mesh_z,mesh_r) #For mesh figure : 40x60 double

	dist_initial_ref = density*np.exp(-mesh_Z**2/(2*v_th_z**2) - mesh_R**2/(v_th_r**2)) / (np.sqrt(2*np.pi)*np.pi*(v_th_r**2)*v_th_z) #Bi-Maxwellian

	dist = dist_initial_ref

	numeric_T = mass*(((mesh_R**2 + mesh_Z**2)*dist*vol).sum(axis=0)).sum(axis=0) / (3*echarge*density)
	T_overall = numeric_T
	Coulomb_log= 23 - np.log(np.sqrt(2*density*1e-6/T_overall)/T_overall)
	col_factor = echarge**4 * Coulomb_log / ( 8 * np.pi * eps_0**2 * mass**2 )
	time_reference = 8 * np.pi * np.sqrt(2*mass) * eps_0*eps_0 * (T_overall*echarge)**1.5 / (density* echarge**4 * Coulomb_log)
	tstep = t_hat_step * time_reference

	diag_params = {'tstep': tstep, 'vol': vol, 'mesh_Z': mesh_Z, 'mesh_R': mesh_R}

	return diag_params

def WTdiagnosis(diff_T_factor):

	device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

	mesh_Nr = 40
	mesh_Nz = 60   # Generally Nz = Nr*2 due to different domain length
	density = 10**19
	vol_real = 0.1
	echarge = 1.6022e-19  # Electron Charge
	mass = 1.6276e-27 # Kg
	eps_0 = 8.8542e-12
	Tr = 100 # Temperature unit : eV
	Tz = Tr * float(diff_T_factor)  # Note that Tr = 1/2(Tx+Ty), i.e. Equilibrium : Tx=Ty=Tz=Tr
	t_hat_step = 0.1   #Units : time_reference

	# System Parameter
	v_th_z = np.sqrt(Tz*echarge/mass)
	v_th_r = np.sqrt(2*Tr*echarge/mass)
	mesh_bd_factor = 1.05 # boundary = mesh_bd_factor * maximum velocity of particle ; 1.1

	bd_mesh_factor = 5 # unit: v_th
	vz_max = v_th_z * bd_mesh_factor
	vz_min = -vz_max
	vr_max = v_th_r * bd_mesh_factor

	mesh_zN = np.max([np.abs(vz_max), np.abs(vz_min)])*mesh_bd_factor
	mesh_z1 = -mesh_zN
	mesh_rN = vr_max*mesh_bd_factor
	mesh_Nrm1 = mesh_Nr-1
	mesh_r = np.linspace(0,mesh_rN,mesh_Nr)
	mesh_z = np.linspace(mesh_z1,mesh_zN,mesh_Nz)

	mesh_dr = mesh_r[1]-mesh_r[0]
	mesh_dz = mesh_z[1]-mesh_z[0]

	[mesh_Z, mesh_R]=np.meshgrid(mesh_z,mesh_r) #For mesh figure : 40x60 double

	dist_initial_ref = density*np.exp(-mesh_Z**2/(2*v_th_z**2) - mesh_R**2/(v_th_r**2)) / (np.sqrt(2*np.pi)*np.pi*(v_th_r**2)*v_th_z) #Bi-Maxwellian

	dist = dist_initial_ref

	numeric_T = mass*(((mesh_R**2 + mesh_Z**2)*dist*vol).sum(axis=0)).sum(axis=0) / (3*echarge*density)
	T_overall = numeric_T
	Coulomb_log= 23 - np.log(np.sqrt(2*density*1e-6/T_overall)/T_overall)
	col_factor = echarge**4 * Coulomb_log / ( 8 * np.pi * eps_0**2 * mass**2 )
	time_reference = 8 * np.pi * np.sqrt(2*mass) * eps_0*eps_0 * (T_overall*echarge)**1.5 / (density* echarge**4 * Coulomb_log)
	tstep = t_hat_step * time_reference

	fn_totorch = lambda x: torch.Tensor(x).to(device)

	vol = fn_totorch(vol)
	mesh_Z = fn_totorch(mesh_Z)
	mesh_R = fn_totorch(mesh_R)

	diag_params = {'tstep': tstep, 'vol': vol, 'mesh_Z': mesh_Z, 'mesh_R': mesh_R, 'mesh_dr': mesh_dr, 'mesh_dz': mesh_dz}

	return diag_params

def Conservative_loss(weight_E, weight_D, weight_M, label_filename, label, output, label_ori, input):
	"""
	label, output : df, label_ori, input: f
	Caution: weight_E is devided but weight_D and weight_M are multiplied to each terms.
	return: E_loss->E squared loss,
	E_error, Target_E_loss,	D_loss, Target_D_loss,	M_loss,	Target_M_loss->each physical loss
	"""
	E_loss_target = []
	E_loss = []
	E_error = []
	D_loss_target = []
	D_loss = []
	M_loss_target = []
	M_loss = []

	for j in range(label.shape[0]):
		diff_T_factor = label_filename[j][6:14]
		diag_params = diagnosis(diff_T_factor)
		tstep, vol, mesh_Z, mesh_R = diag_params['tstep'], diag_params['vol'], diag_params['mesh_Z'], diag_params['mesh_R']
		echarge = 1.6022e-19  # Electron Charge
		mass = 1.6276e-27 # Kg
		Tr = 100 # Temperature unit : eV
		Tz = Tr * float(diff_T_factor)  # Note that Tr = 1/2(Tx+Ty), i.e. Equilibrium : Tx=Ty=Tz=Tr
		v_th_z = np.sqrt(Tz*echarge/mass)
		v_th_r = np.sqrt(2*Tr*echarge/mass)
		
		dist_iter = output[j].squeeze()
		dist_iter_label = label[j].squeeze()
		dist_iter_label_ori = label_ori[j].squeeze()
		dist_iter_input = input[j].squeeze()

		# output - label
		dFdt_inner = (dist_iter - dist_iter_label)
		dFdtc_inner = dFdt_inner*vol
		# target label - input
		dFdt_target = (dist_iter_label_ori - dist_iter_input)
		dFdtc_target = dFdt_target*vol

		# dwdt = energy, dndt = particle number
		dndt = ((dFdtc_inner).sum(axis=0)).sum(axis=0)
		dwdt = (dFdtc_inner * (mesh_Z**2 + mesh_R**2))
		E_error_dwdt = dwdt
		E_error_dwdt_sum = ((E_error_dwdt).sum(axis=0)).sum(axis=0)
		Wdwdt = dwdt/weight_E
		Tdwdt = Wdwdt.mul(Wdwdt) # E squared loss
		dwdt_sum = ((Tdwdt).sum(axis=0)).sum(axis=0)

		# target
		dndt_target = ((dFdtc_target).sum(axis=0)).sum(axis=0)
		dwdt_target = dFdtc_target * (mesh_Z**2 + mesh_R**2)
		dwdt_sum_target = ((dwdt_target).sum(axis=0)).sum(axis=0)

		# density calculate
		diag_density_label = dist_iter_label_ori * vol
		diag_density_input = dist_iter_input * vol

		# momentum calculate
		dM_Z = dFdt_inner * mesh_Z * vol
		dM_Z = (dM_Z.sum(axis=0)).sum(axis=0)
		dM_Z_target = dFdt_target * mesh_Z * vol
		dM_Z_target = (dM_Z_target.sum(axis=0)).sum(axis=0)
		v_th = v_th_z + v_th_r

		dM_R_target = dFdt_target * mesh_R * vol
		dM_R_target = (dM_R_target.sum(axis=0)).sum(axis=0)

		# Diagnostic
		Eenergy_error = E_error_dwdt_sum/((dist_iter_label_ori*(mesh_Z**2 + mesh_R**2)*vol).sum(axis=0)).sum(axis=0)
		diag_Energy_error = dwdt_sum/((dist_iter_label_ori*(mesh_Z**2 + mesh_R**2)*vol).sum(axis=0)).sum(axis=0)
		diag_Energy_error_target = dwdt_sum_target/((dist_iter_input*(mesh_Z**2 + mesh_R**2)*vol).sum(axis=0)).sum(axis=0)
		diag_Density_error = dndt/((diag_density_label).sum(axis=0)).sum(axis=0)
		diag_Density_error_target = dndt_target/((diag_density_input).sum(axis=0)).sum(axis=0)
		diag_Momentum_error = (dM_Z)/(((diag_density_label).sum(axis=0)).sum(axis=0)*v_th)
		diag_Momentum_error_target = (dM_Z_target)/(((diag_density_input).sum(axis=0)).sum(axis=0)*v_th)

		E_error += [Eenergy_error]
		E_loss += [diag_Energy_error]
		E_loss_target += [diag_Energy_error_target]
		D_loss += [diag_Density_error]
		D_loss_target += [diag_Density_error_target]
		M_loss += [diag_Momentum_error]
		M_loss_target += [diag_Momentum_error_target]

	E_error = torch.abs(torch.mean(torch.stack(E_error)))
	E_loss = torch.abs(torch.mean(torch.stack(E_loss)))
	Target_E_loss = torch.abs(torch.mean(torch.stack(E_loss_target)))
	D_loss = torch.abs(torch.mean(torch.stack(D_loss)))
	Target_D_loss = torch.abs(torch.mean(torch.stack(D_loss_target)))
	M_loss = torch.abs(torch.mean(torch.stack(M_loss)))
	Target_M_loss = torch.abs(torch.mean(torch.stack(M_loss_target)))

	phy_loss = {'E_loss': E_loss, 'E_error': E_error, 'Target_E_loss': Target_E_loss,'D_loss': D_loss, 'Target_D_loss': Target_D_loss,'M_loss': M_loss, 'Target_M_loss': Target_M_loss}

	return phy_loss

def WT_Conservative_loss(weight_E, weight_D, weight_M, label_filename, label, output, label_ori, input):
	E_loss_target = []
	E_loss = []
	E_error = []
	D_loss_target = []
	D_loss = []
	M_loss_target = []
	M_loss = []
	TZ = []
	TR = []
	TZ_target = []
	TR_target = []

	for j in range(label.shape[0]):
		diff_T_factor = label_filename[j][6:14]
		diag_params = WTdiagnosis(diff_T_factor)
		tstep, vol, mesh_Z, mesh_R, mesh_dr, mesh_dz = diag_params['tstep'], diag_params['vol'], diag_params['mesh_Z'], diag_params['mesh_R'], diag_params['mesh_dr'], diag_params['mesh_dz']
		echarge = 1.6022e-19  # Electron Charge
		mass = 1.6276e-27 # Kg
		Tr = 100 # Temperature unit : eV
		Tz = Tr * float(diff_T_factor)  # Note that Tr = 1/2(Tx+Ty), i.e. Equilibrium : Tx=Ty=Tz=Tr
		v_th_z = np.sqrt(Tz*echarge/mass)
		v_th_r = np.sqrt(2*Tr*echarge/mass)
		
		dist_iter = output[j].squeeze()
		dist_iter_label = label[j].squeeze()
		dist_iter_label_ori = label_ori[j].squeeze()
		dist_iter_input = input[j].squeeze()

		# output - label
		dFdt_inner = (dist_iter - dist_iter_label)
		dFdtc_inner = dFdt_inner*vol
		# target label - input
		dFdt_target = (dist_iter_label_ori - dist_iter_input)
		dFdtc_target = dFdt_target*vol

		# dwdt = energy, dndt = particle number
		dndt = ((dFdtc_inner).sum(axis=0)).sum(axis=0)
		dwdt = (dFdtc_inner * (mesh_Z**2 + mesh_R**2))
		E_error_dwdt = dwdt
		E_error_dwdt_sum = ((E_error_dwdt).sum(axis=0)).sum(axis=0)
		Wdwdt = dwdt/weight_E
		Tdwdt = Wdwdt.mul(Wdwdt) # E squared loss
		dwdt_sum = ((Tdwdt).sum(axis=0)).sum(axis=0)
		# Temperature of each axis
		mesh_dz = torch.ones((40,60), device='cuda:0')*mesh_dz
		mesh_dr = torch.ones((40,60), device='cuda:0')*mesh_dr
		diag_Energy_Z = 0.5*mass*(((dist_iter+dist_iter_input)*(mesh_Z**2+mesh_dz**2/12)*vol).sum(axis=0)).sum(axis=0)
		diag_Energy_R = 0.5*mass*(((dist_iter+dist_iter_input)*(mesh_R**2+mesh_dr**2/4)*vol).sum(axis=0)).sum(axis=0)
		diag_T_Z = 2*diag_Energy_Z/(echarge*1.e+19)
		diag_T_R = diag_Energy_R/(echarge*1.e+19)
		diag_Energy_Z_target = 0.5*mass*((dist_iter_label_ori*(mesh_Z**2+mesh_dz**2/12)*vol).sum(axis=0)).sum(axis=0)
		diag_Energy_R_target = 0.5*mass*((dist_iter_label_ori*(mesh_R**2+mesh_dr**2/4)*vol).sum(axis=0)).sum(axis=0)
		diag_T_Z_target = 2*diag_Energy_Z_target/(echarge*1.e+19)
		diag_T_R_target =diag_Energy_R_target/(echarge*1.e+19)
	
		# target
		dndt_target = ((dFdtc_target).sum(axis=0)).sum(axis=0)
		dwdt_target = dFdtc_target * (mesh_Z**2 + mesh_R**2)
		dwdt_sum_target = ((dwdt_target).sum(axis=0)).sum(axis=0)

		# density calculate
		diag_density_label = dist_iter_label_ori * vol
		diag_density_input = dist_iter_input * vol

		# momentum calculate
		dM_Z = dFdt_inner * mesh_Z * vol
		dM_Z = (dM_Z.sum(axis=0)).sum(axis=0)

		dM_Z_target = dFdt_target * mesh_Z * vol
		dM_Z_target = (dM_Z_target.sum(axis=0)).sum(axis=0)
		v_th = v_th_z + v_th_r

		dM_R_target = dFdt_target * mesh_R * vol
		dM_R_target = (dM_R_target.sum(axis=0)).sum(axis=0)

		# Diagnostic
		Eenergy_error = E_error_dwdt_sum/((dist_iter_label_ori*(mesh_Z**2 + mesh_R**2)*vol).sum(axis=0)).sum(axis=0)
		diag_Energy_error = dwdt_sum/((dist_iter_label_ori*(mesh_Z**2 + mesh_R**2)*vol).sum(axis=0)).sum(axis=0)
		diag_Energy_error_target = dwdt_sum_target/((dist_iter_input*(mesh_Z**2 + mesh_R**2)*vol).sum(axis=0)).sum(axis=0)
		diag_Density_error = dndt/((diag_density_label).sum(axis=0)).sum(axis=0)
		diag_Density_error_target = dndt_target/((diag_density_input).sum(axis=0)).sum(axis=0)
		diag_Momentum_error = (dM_Z)/(((diag_density_label).sum(axis=0)).sum(axis=0)*v_th)
		diag_Momentum_error_target = (dM_Z_target)/(((diag_density_input).sum(axis=0)).sum(axis=0)*v_th)

		E_error += [Eenergy_error]
		E_loss += [diag_Energy_error]
		E_loss_target += [diag_Energy_error_target]
		D_loss += [diag_Density_error]
		D_loss_target += [diag_Density_error_target]
		M_loss += [diag_Momentum_error]
		M_loss_target += [diag_Momentum_error_target]
	
		# Temperatuire
		TZ += [diag_T_Z]
		TR += [diag_T_R]
		TZ_target += [diag_T_Z_target]
		TR_target += [diag_T_R_target]

	E_error = torch.abs(torch.mean(torch.stack(E_error)))
	E_loss = torch.abs(torch.mean(torch.stack(E_loss)))
	Target_E_loss = torch.abs(torch.mean(torch.stack(E_loss_target)))
	D_loss = torch.abs(torch.mean(torch.stack(D_loss)))
	Target_D_loss = torch.abs(torch.mean(torch.stack(D_loss_target)))
	M_loss = torch.abs(torch.mean(torch.stack(M_loss)))
	Target_M_loss = torch.abs(torch.mean(torch.stack(M_loss_target)))

	phy_loss = {'E_loss': E_loss, 'E_error': E_error, 'Target_E_loss': Target_E_loss,'D_loss': D_loss, 'Target_D_loss': Target_D_loss,'M_loss': M_loss, 'Target_M_loss': Target_M_loss, 'TZ': TZ, 'TR': TR, 'TZ_target': TZ_target, 'TR_target': TR_target}

	return phy_loss