import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math

import src.numpy_utility as pnu
from src.file_utility import save_stuff, flatten_dict, embed_dict

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim

from torchmodel.models.alexnet import Alexnet_fmaps

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 

def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)        

class Torch_fwRF_voxel_block(nn.Module):

    def __init__(self, _fmaps_fn, params, _nonlinearity=None, input_shape=(1,3,227,227), aperture=1.0, device=torch.device("cpu")):
        super(Torch_fwRF_voxel_block, self).__init__()
        
        self.aperture = aperture
        models, weights, bias, mstmt, mstst = params
        _x = torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
        
        self.pfs = []
        for k,n_pix in enumerate(self.fmaps_rez):
            pf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            self.pfs += [nn.Parameter(torch.from_numpy(pf).to(device), requires_grad=False),]
            self.register_parameter('pf%d'%k, self.pfs[-1])
            
        self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
            
        self.mstm = None
        self.msts = None
        if mstmt is not None:
            self.mstm = nn.Parameter(torch.from_numpy(mstmt.T).to(device), requires_grad=False)
        if mstst is not None:
            self.msts = nn.Parameter(torch.from_numpy(mstst.T).to(device), requires_grad=False)
        self._nl = _nonlinearity
              
    def load_voxel_block(self, *params):
        models = params[0]
        for _pf,n_pix in zip(self.pfs, self.fmaps_rez):
            pf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=self.aperture, dtype=np.float32)[2]
            if len(pf)<_pf.size()[0]:
                pp = np.zeros(shape=_pf.size(), dtype=pf.dtype)
                pp[:len(pf)] = pf
                set_value(_pf, pp)
            else:
                set_value(_pf, pf)
        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp)
                else:
                    set_value(_p, p)
        for _p,p in zip([self.mstm, self.msts], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp.T)
                else:
                    set_value(_p, p.T)
 
    def forward(self, _fmaps):
        _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [1,2]]) for _fm,_pf in zip(_fmaps, self.pfs)], dim=1) # [#samples, #features, #voxels] 
        if self._nl is not None:
            _mst = self._nl(_mst)
        if self.mstm is not None:              
            _mst -= self.mstm[None]
        if self.msts is not None:
            _mst /= self.msts[None]
        _mst = torch.transpose(torch.transpose(_mst, 0, 2), 1, 2) # [#voxels, #samples, features]
        _r = torch.squeeze(torch.bmm(_mst, torch.unsqueeze(self.weights, 2))).t() # [#samples, #voxels]
        if self.bias is not None:
            _r += self.bias
        return _r

class Torch_filter_fmaps(nn.Module):
    def __init__(self, _fmaps, lmask, fmask):
        super(Torch_filter_fmaps, self).__init__()
        device = next(_fmaps.parameters()).device
        self.fmaps = _fmaps
        self.lmask = lmask
        self.fmask = [nn.Parameter(torch.from_numpy(fm).to(device), requires_grad=False) for fm in fmask]
        for k,fm in enumerate(self.fmask):
             self.register_parameter('fm%d'%k, fm)

    def forward(self, _x):
        _fmaps = self.fmaps(_x)
        return [torch.index_select(torch.cat([_fmaps[l] for l in lm], 1), dim=1, index=fm) for lm,fm in zip(self.lmask, self.fmask)]


def load_encoding(subject, model_name='dnn_fwrf',device=torch.device("cpu")):
    
    voxel_batch_size = 24
    
    root_dir = "./"
    output_dir = root_dir + "output/S%02d/%s/" % (subject,model_name) 
    model_params_set = h5py.File(output_dir + 'model_params.h5py', 'r')
    model_params = embed_dict({k: np.copy(d) for k,d in model_params_set.items()})
    model_params_set.close()
    
    _fmaps_fn = Alexnet_fmaps().to(device)
    _fmaps_fn = Torch_filter_fmaps(_fmaps_fn, model_params['lmask'], model_params['fmask'])
    
    params = [p[:voxel_batch_size] if p is not None else None for p in model_params['params']]
    
    _fwrf_fn  = Torch_fwRF_voxel_block(_fmaps_fn, params, _nonlinearity=None, input_shape=(1,3,227,227), aperture=1.0, device=device)
    with torch.no_grad():
        _fwrf_fn.load_voxel_block(*[p[0:voxel_batch_size] if p is not None else None for p in model_params['params']])
    
    
    #device = next(_fmaps_fn.parameters()).device
    #_params = [_p for _p in _fwrf_fn.parameters()]
    #voxel_batch_size = _params[0].size()[0]    
    #nt, nv = len(data), len(params[0])
    
    return _fwrf_fn, _fmaps_fn
    