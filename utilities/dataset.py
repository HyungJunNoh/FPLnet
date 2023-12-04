# Develped by Hyeongjun Noh - submission for Nature communication, December, 2023.
# This code is private and not intended for public distribution. 
# Any unauthorized actions including sharing, distribution, or modification without the explicit permission of the author are strictly prohibited.
# Hyeongjun Noh nhj12074@unist.ac.kr, Jimin Lee jiminlee@unist.ac.kr, Eisung Yoon esyoon@unist.ac.kr
import os
import numpy as np
import torch
from utilities.util import *

torch.set_default_dtype(torch.float64)
# Data loader
class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.transform = transform
    
    lst_data = os.listdir(self.data_dir)

    lst_label = [f for f in lst_data if f.startswith('label')]
    lst_input = [f for f in lst_data if f.startswith('input')]

    lst_label.sort()
    lst_input.sort()

    self.lst_label = lst_label
    self.lst_input = lst_input

  def __len__(self):
    return len(self.lst_label)

  def __getitem__(self, index):
    label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
    input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
    label_filename = self.lst_label[index]
    input_filename = self.lst_input[index]
    
    if label.ndim == 2:
      label = label[:, :, np.newaxis]
    if input.ndim == 2:
      input = input[:, :, np.newaxis]

    diff_T_factor = label_filename[6:14]
    diag_params = diagnosis_noTorch(diff_T_factor)
    tstep, vol, mesh_Z, mesh_R = diag_params['tstep'], diag_params['vol'], diag_params['mesh_Z'], diag_params['mesh_R']

    mesh_Z = mesh_Z[:,:,np.newaxis]/1e+10
    mesh_R = mesh_R[:,:,np.newaxis]/1e+10
    input = np.concatenate((input, mesh_R, mesh_Z), axis=2)

    data = {'input': input, 'label': label, 'input_filename': input_filename, 'label_filename': label_filename}

    if self.transform:
      data = self.transform(data)

    return data

# Data transform
class ToTensor(object):
  def __call__(self, data):
    label, input, input_filename, label_filename = data['label'], data['input'], data['input_filename'], data['label_filename']

    label = label.transpose((2, 0, 1)).astype(np.float64) # numpy(Y, X, CH) -> (CH, Y, X)
    input = input.transpose((2, 0, 1)).astype(np.float64)

    data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input), 'input_filename': input_filename, 'label_filename': label_filename}

    return data

class Normalization(object):
  def __init__(self, min=0, max=770.0):
    self.min = min
    self.max = max

  def __call__(self, data):
    label, input, input_filename, label_filename = data['label'], data['input'], data['input_filename'], data['label_filename']

    input[:,:,0] = (input[:,:,0].astype(np.float64) - self.min) / (self.max - self.min)

    data = {'label': label, 'input': input, 'input_filename': input_filename, 'label_filename': label_filename}
    #data = {'input': input}

    return data

class RandomFlip(object):
  def __call__(self, data):
    label, input, input_filename, label_filename = data['label'], data['input'], data['input_filename'], data['label_filename']

    if np.random.rand() > 0.7:
      label = np.fliplr(label)
      input = np.fliplr(input)  

    data = {'label': label, 'input': input, 'input_filename': input_filename, 'label_filename': label_filename}

    return data
