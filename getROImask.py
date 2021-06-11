import sys
import os
import struct
import time as time
import numpy as np
import h5py
from scipy.stats import pearsonr
from itertools import chain
from scipy.io import loadmat
import pickle
import math
import matplotlib.pyplot as plt

#import argparse
import csv
from itertools import zip_longest

from src.file_utility import load_mask_from_nii, view_data

nsd_root = "/home/zg243/nsd/"
stim_root = nsd_root + "stimuli/"
beta_root = nsd_root + "responses/"
mask_root = nsd_root + "mask/ppdata/"
roi_root = nsd_root + "freesurfer/"

exp_design_file = nsd_root + "experiments/nsd_expdesign.mat"
stim_file       = stim_root + "nsd_stimuli.hdf5"

subject = 8
roimask_dir = nsd_root + 'roimask/subj%02d/'%subject
# load masks
tight_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%subject) 
brain_mask_full = tight_mask_full.flatten().astype(bool)
voxel_idx_brain = np.arange(len(brain_mask_full))[brain_mask_full]

# load the ROI files
# floc_faces  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz"%(subject))
# floc_bodies = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz"%(subject))
# floc_places = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz"%(subject))
# floc_words  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz"%(subject))
# visualrois  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%(subject))

# load amygdala and hippocampus
aseg = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/aseg.nii.gz"%(subject))

# OFA_mask, FFA1_mask, FFA2_mask, mTLfaces_mask, aTLfaces_mask = np.array(floc_faces), np.array(floc_faces), np.array(floc_faces), np.array(floc_faces), np.array(floc_faces)
# EBA_mask, FBA1_mask, FBA2_mask, mTLbodies_mask = np.array(floc_bodies), np.array(floc_bodies), np.array(floc_bodies), np.array(floc_bodies)
# OPA_mask, PPA_mask, RSC_mask = np.array(floc_places), np.array(floc_places), np.array(floc_places)
# OWFA_mask, VWFA1_mask, VWFA2_mask, mfswords_mask, mTLwords_mask = np.array(floc_words), np.array(floc_words), np.array(floc_words), np.array(floc_words), np.array(floc_words)
# V1v_mask, V1d_mask, V2v_mask, V2d_mask, V3v_mask, V3d_mask, hV4_mask = np.array(visualrois), np.array(visualrois), np.array(visualrois), np.array(visualrois), np.array(visualrois), np.array(visualrois), np.array(visualrois)

Lhipp_mask, Lamy_mask, Rhipp_mask, Ramy_mask = np.array(aseg), np.array(aseg), np.array(aseg), np.array(aseg) 

# OFA_mask[floc_faces != 1]        = 0
# OFA_mask[floc_faces == 1]        = 1
# FFA1_mask[floc_faces != 2]       = 0
# FFA1_mask[floc_faces == 2]       = 1
# FFA2_mask[floc_faces != 3]       = 0
# FFA2_mask[floc_faces == 3]       = 1
# mTLfaces_mask[floc_faces != 4]   = 0
# mTLfaces_mask[floc_faces == 4]   = 1
# aTLfaces_mask[floc_faces != 5]   = 0
# aTLfaces_mask[floc_faces == 5]   = 1

# EBA_mask[floc_bodies != 1]       = 0
# EBA_mask[floc_bodies == 1]       = 1
# FBA1_mask[floc_bodies != 2]      = 0
# FBA1_mask[floc_bodies == 2]      = 1
# FBA2_mask[floc_bodies != 3]      = 0
# FBA2_mask[floc_bodies == 3]      = 1
# mTLbodies_mask[floc_bodies != 4] = 0
# mTLbodies_mask[floc_bodies == 4] = 1

# OPA_mask[floc_places != 1]       = 0
# OPA_mask[floc_places == 1]       = 1
# PPA_mask[floc_places != 2]       = 0
# PPA_mask[floc_places == 2]       = 1
# RSC_mask[floc_places != 3]       = 0
# RSC_mask[floc_places == 3]       = 1

# OWFA_mask[floc_words != 1]       = 0
# OWFA_mask[floc_words == 1]       = 1
# VWFA1_mask[floc_words != 2]      = 0
# VWFA1_mask[floc_words == 2]      = 1
# VWFA2_mask[floc_words != 3]      = 0
# VWFA2_mask[floc_words == 3]      = 1
# mfswords_mask[floc_words != 4]   = 0
# mfswords_mask[floc_words == 4]   = 1
# mTLwords_mask[floc_words != 5]   = 0
# mTLwords_mask[floc_words == 5]   = 1

# V1v_mask[visualrois != 1] = 0
# V1v_mask[visualrois == 1] = 1
# V1d_mask[visualrois != 2] = 0
# V1d_mask[visualrois == 2] = 1
# V2v_mask[visualrois != 3] = 0
# V2v_mask[visualrois == 3] = 1
# V2d_mask[visualrois != 4] = 0
# V2d_mask[visualrois == 4] = 1
# V3v_mask[visualrois != 5] = 0
# V3v_mask[visualrois == 5] = 1
# V3d_mask[visualrois != 6] = 0
# V3d_mask[visualrois == 6] = 1
# hV4_mask[visualrois != 7] = 0
# hV4_mask[visualrois == 7] = 1

Lhipp_mask[aseg != 17] = 0
Lhipp_mask[aseg == 17] = 1
Lamy_mask[aseg != 18]  = 0
Lamy_mask[aseg == 18]  = 1
Rhipp_mask[aseg != 53] = 0
Rhipp_mask[aseg == 53] = 1
Ramy_mask[aseg != 54]  = 0
Ramy_mask[aseg == 54]  = 1

# OFAmask_flatten       = OFA_mask.flatten()[brain_mask_full]
# FFA1mask_flatten      = FFA1_mask.flatten()[brain_mask_full]
# FFA2mask_flatten      = FFA2_mask.flatten()[brain_mask_full]
# mTLfacesmask_flatten  = mTLfaces_mask.flatten()[brain_mask_full]
# aTLfacesmask_flatten  = aTLfaces_mask.flatten()[brain_mask_full]

# EBAmask_flatten       = EBA_mask.flatten()[brain_mask_full]
# FBA1mask_flatten      = FBA1_mask.flatten()[brain_mask_full]
# FBA2mask_flatten      = FBA2_mask.flatten()[brain_mask_full]
# mTLbodiesmask_flatten = mTLbodies_mask.flatten()[brain_mask_full]

# OPAmask_flatten       = OPA_mask.flatten()[brain_mask_full]
# PPAmask_flatten       = PPA_mask.flatten()[brain_mask_full]
# RSCmask_flatten       = RSC_mask.flatten()[brain_mask_full]
 
# OWFAmask_flatten      = OWFA_mask.flatten()[brain_mask_full]
# VWFA1mask_flatten     = VWFA1_mask.flatten()[brain_mask_full]
# VWFA2mask_flatten     = VWFA2_mask.flatten()[brain_mask_full]
# mfswordsmask_flatten  = mfswords_mask.flatten()[brain_mask_full]
# mTLwordsmask_flatten  = mTLwords_mask.flatten()[brain_mask_full]

# V1vmask_flatten = V1v_mask.flatten()[brain_mask_full]
# V1dmask_flatten = V1d_mask.flatten()[brain_mask_full]
# V2vmask_flatten = V2v_mask.flatten()[brain_mask_full]
# V2dmask_flatten = V2d_mask.flatten()[brain_mask_full]
# V3vmask_flatten = V3v_mask.flatten()[brain_mask_full]
# V3dmask_flatten = V3d_mask.flatten()[brain_mask_full]
# hV4mask_flatten = hV4_mask.flatten()[brain_mask_full] 

Lhippmask_flatten = Lhipp_mask.flatten()[brain_mask_full]
Lamymask_flatten  = Lamy_mask.flatten()[brain_mask_full]
Rhippmask_flatten = Rhipp_mask.flatten()[brain_mask_full]
Ramymask_flatten  = Ramy_mask.flatten()[brain_mask_full]

# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, OFAmask_flatten, save_to=roimask_dir+'OFA')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, FFA1mask_flatten, save_to=roimask_dir+'FFA1')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, FFA2mask_flatten, save_to=roimask_dir+'FFA2')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, mTLfacesmask_flatten, save_to=roimask_dir+'mTLfaces')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, aTLfacesmask_flatten, save_to=roimask_dir+'aTLfaces')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, EBAmask_flatten, save_to=roimask_dir+'EBA')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, FBA1mask_flatten, save_to=roimask_dir+'FBA1')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, FBA2mask_flatten, save_to=roimask_dir+'FBA2')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, mTLbodiesmask_flatten, save_to=roimask_dir+'mTLbodies')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, OPAmask_flatten, save_to=roimask_dir+'OPA')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, PPAmask_flatten, save_to=roimask_dir+'PPA')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, RSCmask_flatten, save_to=roimask_dir+'RSC')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, OWFAmask_flatten, save_to=roimask_dir+'OWFA')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, VWFA1mask_flatten, save_to=roimask_dir+'VWFA1')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, VWFA2mask_flatten, save_to=roimask_dir+'VWFA2')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, mfswordsmask_flatten, save_to=roimask_dir+'mfswords')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, mTLwordsmask_flatten, save_to=roimask_dir+'mTLwords')

# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, V1vmask_flatten, save_to=roimask_dir+'V1v')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, V1dmask_flatten, save_to=roimask_dir+'V1d')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, V2vmask_flatten, save_to=roimask_dir+'V2v')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, V2dmask_flatten, save_to=roimask_dir+'V2d')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, V3vmask_flatten, save_to=roimask_dir+'V3v')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, V3dmask_flatten, save_to=roimask_dir+'V3d')
# volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, hV4mask_flatten, save_to=roimask_dir+'hV4')

volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, Lhippmask_flatten, save_to=roimask_dir+'L_hippocampus')
volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, Lamymask_flatten,  save_to=roimask_dir+'L_amygdala')
volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, Rhippmask_flatten, save_to=roimask_dir+'R_hippocampus')
volume_brain_mask = view_data(tight_mask_full.shape, voxel_idx_brain, Ramymask_flatten,  save_to=roimask_dir+'R_amygdala')









