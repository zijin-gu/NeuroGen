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
#import argparse
import scipy.io as sio
from itertools import zip_longest

from src.file_utility import load_mask_from_nii, view_data
from src.load_nsd import load_betas

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
beta_root = nsd_root + "responses/"
mask_root = nsd_root + "mask/ppdata/"
roi_root = nsd_root + "freesurfer/"
roimask_dir = nsd_root + 'roimask/'
roibeta_dir = nsd_root + 'roibeta/'

exp_design_file = nsd_root + "experiments/nsd_expdesign.mat"
stim_file       = stim_root + "nsd_stimuli.hdf5"

subject = 8
ROIs = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']

tight_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%subject) 
brain_mask_full = tight_mask_full.flatten().astype(bool)
voxel_idx_brain = np.arange(len(brain_mask_full))[brain_mask_full]

voxel_mask = brain_mask_full
voxel_idx = voxel_idx_brain

beta_subj = beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/" % (subject,)
voxel_data, filenames = load_betas(folder_name=beta_subj, zscore=True, voxel_mask=voxel_mask, up_to=-1, load_ext='.nii.gz')
print ("voxel data shape is: ", voxel_data.shape) # (22500, 234817)

# load ROI masks
for roi in ROIs:
	roimask = load_mask_from_nii(roimask_dir + 'subj%02d/'%(subject) + roi + '.nii')
	roimask_flatten = roimask.flatten()[brain_mask_full]

	# apply ROI mask to shared voxel data
	masked_voxel_data = voxel_data[:, roimask_flatten != 0]

	# calculate the mean of the activations in the ROI, which is a (22500,) vector
	#voxel_num = np.where(roimask_flatten != 0)[0].shape[0]
	#mean_activation = np.sum(masked_voxel_data, axis=1)/voxel_num

	#np.savetxt(roibeta_dir + "subj%02d/meanbeta_"%subject + roi + ".txt", mean_activation)

	mdic = {"beta": masked_voxel_data}
	sio.savemat(roibeta_dir + 'subj%02d/beta_' + roi + '.mat', mdic)

	del masked_voxel_data
	print("ROI " + roi + " is finished!")

