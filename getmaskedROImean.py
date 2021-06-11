import sys
import os
import struct
import time as time
import numpy as np
import h5py
from scipy.stats import pearsonr
from itertools import chain
from scipy.io import loadmat
#from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
#import seaborn as sns
import csv
from itertools import zip_longest

from src.file_utility import load_mask_from_nii, view_data
from src.load_nsd import load_betas
import argparse

parser = argparse.ArgumentParser(prog='get ROI mean for each subject', 
	description='input subject ID, output txt files',
	usage='python getmaskedROImean.py --subj i')

parser.add_argument('--subj', type=int)
args = parser.parse_args()

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
beta_root = nsd_root + "responses/"
mask_root = nsd_root + "mask/ppdata/"
roi_root = nsd_root + "freesurfer/"
roimask_dir = nsd_root + 'roimask/'
roibeta_dir = nsd_root + 'roiavgbeta/'

exp_design_file = nsd_root + "experiments/nsd_expdesign.mat"
stim_file       = stim_root + "nsd_stimuli.hdf5"

subject = args.subj

# ROIs = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs = ['L_hippocampus', 'L_amygdala', 'R_hippocampus', 'R_amygdala']

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
	masked_voxel_data = roimask_flatten * voxel_data

	# calculate the average of the activations in the ROI, which is a (22500,) vector
	voxel_num = np.where(roimask_flatten != 0)[0].shape[0]
	mean_activation = np.sum(masked_voxel_data, axis=1)/voxel_num

	np.savetxt(roibeta_dir + "subj%02d/meanbeta_"%subject + roi + ".txt", mean_activation)
	del masked_voxel_data
	print("ROI " + roi + " is finished!")

