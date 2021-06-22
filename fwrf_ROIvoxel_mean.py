import sys
import os
import struct
import time as time
import numpy as np
import h5py
from scipy.stats import pearsonr
from itertools import chain
from scipy.io import loadmat
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from itertools import zip_longest

fpX = np.float32
import src.numpy_utility as pnu
from src.plots import display_candidate_loss
from src.file_utility import save_stuff, flatten_dict, embed_dict
from src.torch_fwrf import get_value

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import DataLoader,Dataset
import argparse

parser = argparse.ArgumentParser(prog='encoding model', 
	description='input subject ID and gpu, output saved parameters',
	usage='python fwrf_ROIvoxel_mean.py --subj i --gpu j')

parser.add_argument('--subj', type=int)
#parser.add_argument('--gpu', type=int)
args = parser.parse_args()

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
beta_root = nsd_root + "responses/"
mask_root = nsd_root + "mask/ppdata/"
roi_root = nsd_root + "freesurfer/"
meanROIbeta_root = nsd_root + "roiavgbeta/"

exp_design_file = nsd_root + "experiments/nsd_expdesign.mat"
stim_file       = stim_root + "nsd_stimuli.hdf5"

print ('#device:', torch.cuda.device_count())
print ('device#:', torch.cuda.current_device())
print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

torch.manual_seed(time.time())
#device = torch.device("cuda:%d"%args.gpu) #cuda
device = torch.device("cuda") #cuda
torch.backends.cudnn.enabled=True

print ('\ntorch:', torch.__version__)
print ('cuda: ', torch.version.cuda)

subject = args.subj
saveext = ".png"
savearg = {'format':'png', 'dpi': 120, 'facecolor': None}
timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
model_name = 'dnn_fwrf'

root_dir   = os.getcwd() + '/'
net_dir    = root_dir + "net/" 
#input_dir  = '/home/styvesg/repo.data/results/nsd/torch_fwrf_full_brain/S%02d/dnn_fwrf_May-10-2020_1814/'
output_dir = root_dir + "output/S%02d/%s_%s/" % (subject,model_name,timestamp) 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
print ("Time Stamp: %s" % timestamp)

exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py"%subject, 'r')
image_data = np.copy(image_data_set['stimuli']).astype(np.float32) / 255.
image_data_set.close()

print (image_data.shape)
print (image_data.dtype)
print (np.min(image_data[0]), np.max(image_data[0]))

trials = np.array([30000, 30000, 24000, 22500, 30000, 24000, 30000, 22500])
data_size = trials[subject-1]
ordering_data = ordering[:data_size]
shared_mask   = ordering_data<1000  # the first 1000 indices are the shared indices

val_size = np.sum(shared_mask)
trn_size = data_size - val_size
print ("Validation size =", val_size, ", Training size =", trn_size)

stim_data = image_data[ordering_data]  # reduce to only the samples available thus far

trn_stim_data = stim_data[~shared_mask]
val_stim_data = stim_data[shared_mask]

val_image_data = image_data[:1000]

print ('cudnn:', torch.backends.cudnn.version())
print ('dtype:', torch.get_default_dtype())
#torch.set_default_dtype(torch.float64)

ROIs = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
#ROIs = ['L_hippocampus', 'L_amygdala', 'R_hippocampus', 'R_amygdala']
roi_num = len(ROIs)
roi_data = np.zeros([data_size, roi_num])
n = 0
del_idx = []
for roi in ROIs:
    roi_data[:,n] = np.genfromtxt(meanROIbeta_root + 'subj%02d/meanbeta_'%subject + roi + '.txt')
    if np.isnan(np.sum(roi_data[:,n])):
    	del_idx.append(n)
    n += 1

roi_data = np.delete(roi_data, del_idx, axis=1)

trn_roi_data = roi_data[~shared_mask]
val_roi_data = roi_data[shared_mask]

from torchmodel.models.alexnet import Alexnet_fmaps
_fmaps_fn = Alexnet_fmaps().to(device)

_x = torch.tensor(image_data[:100]).to(device) # the input variable.
_fmaps = _fmaps_fn(_x)
for k,_fm in enumerate(_fmaps):
    print (_fm.size())

from src.torch_feature_space import filter_dnn_feature_maps

# I used image_data because the repeats are not relevant
_fmaps_fn, lmask, fmask, tuning_masks = filter_dnn_feature_maps(image_data, _fmaps_fn, batch_size=100, fmap_max=512)

_x = torch.tensor(image_data[:100]).to(device) # the input variable.
_fmaps = _fmaps_fn(_x)
for k,_fm in enumerate(_fmaps):
    print (_fm.size())

from src.rf_grid    import linspace, logspace, model_space, model_space_pyramid
from src.torch_fwrf import learn_params_ridge_regression, get_predictions

aperture = np.float32(1)
nx = ny = 11
smin, smax = np.float32(0.04), np.float32(0.4)
ns = 8

# sharedModel specification is a list of 3 ranges and 3 callable functor. The reason for this is for a future implementation of dynamic mesh refinement.
#model_specs = [[(0., aperture*1.1), (0., aperture*1.1), (smin, smax)], [linspace(nx), linspace(ny), logspace(ns)]]
#models = model_space(model_specs)
models = model_space_pyramid(logspace(ns)(smin, smax), min_spacing=1.4, aperture=1.1*aperture)
print ('candidate count = ', len(models))

sample_batch_size = 200
voxel_batch_size = 500
holdout_size = 3000
lambdas = np.logspace(3.,7.,9, dtype=np.float32)
#_log_act_func = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(torch.abs(_x))

from src.torch_fwrf import  learn_params_ridge_regression, get_predictions, Torch_fwRF_voxel_block

best_losses, best_lambdas, best_params = learn_params_ridge_regression(
    trn_stim_data, trn_roi_data, _fmaps_fn, models, lambdas, \
    aperture=aperture, _nonlinearity=None, zscore=True, sample_batch_size=sample_batch_size, \
    voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=False, add_bias=True)

print ([p.shape if p is not None else None for p in best_params])
param_batch = [p[:voxel_batch_size] if p is not None else None for p in best_params]
_fwrf_fn = Torch_fwRF_voxel_block(_fmaps_fn, param_batch, _nonlinearity=None, input_shape=image_data.shape, aperture=1.0)

voxel_pred = get_predictions(val_image_data, _fmaps_fn, _fwrf_fn, best_params, sample_batch_size=sample_batch_size)

val_voxel_pred = voxel_pred[ordering[:data_size][shared_mask]]
val_cc  = np.zeros(shape=(val_voxel_pred.shape[1]), dtype=fpX)
for v in tqdm(range(val_voxel_pred.shape[1])):    
    val_cc[v] = np.corrcoef(val_roi_data[:,v], val_voxel_pred[:,v])[0,1]  
val_cc = np.nan_to_num(val_cc)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_params = {
    'lmask': lmask,
    'fmask': fmask,
    'tuning_masks': tuning_masks,
    'aperture': aperture,
    #'voxel_mask': voxel_mask,
    #'brain_nii_shape': np.array(brain_nii_shape),
    'val_size': val_size,
    'trn_size': trn_size,
    'shared_mask': shared_mask,
    'image_order': ordering_data,
    #'voxel_index': voxel_idx,
    #'voxel_roi': voxel_roi,
    'params': best_params,
    'lambdas': lambdas, 
    'best_lambdas': best_lambdas,
    'val_cc': val_cc,
    }

print (timestamp)
save_stuff(output_dir + "model_params", flatten_dict(model_params))

ROIs_label = ['OFA', 'FFA1', 'FFA2', 'mTL \n faces', 'aTL \n faces', 'EBA', 'FBA1', 'FBA2', 'mTL \n bodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfs \n words', 'mTL \n words', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
#ROIs_label = ['L_hippocampus', 'L_amygdala', 'R_hippocampus', 'R_amygdala']
ROIs_label = np.delete(ROIs_label,del_idx) 
plt.figure(figsize=(16,6))
plt.bar(ROIs_label,val_cc)
plt.savefig(output_dir + 'acc.png')
