# qtlib.py
#
# Utiliti functions for training CNN in DeepDTI.
#
# (c) Qiyuan Tian, Harvard, 2021

import numpy as np
from tensorflow.keras import backend as K
import torch
import nibabel as nb
def extract_block(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks

def mean_squared_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights # use last channel from grouth-truth data to weight loss from each voxel from first n-1 channels
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return torch.mean(torch.square(y_pred_weighted - y_true_weighted))

def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)
def block_ind(mask, sz_block=64, sz_pad=0):

    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    
    xmin = np.min(xind); xmax = np.max(xind);
    ymin = np.min(yind); ymax = np.max(yind);
    zmin = np.min(zind); zmax = np.max(zind);
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]; 
    
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1;
    ylen = ymax - ymin + 1;
    zlen = zmax - zmin + 1;
    
    nx = int(np.ceil(xlen / sz_block)) + sz_pad;
    ny = int(np.ceil(ylen / sz_block)) + sz_pad;
    nz = int(np.ceil(zlen / sz_block)) + sz_pad;
    
    # determine starting and ending indices of each block
    xstart = xmin;
    ystart = ymin;
    zstart = zmin;
    
    xend = xmax - sz_block + 1;
    yend = ymax - sz_block + 1;
    zend = zmax - sz_block + 1;
    
    xind_block = np.round(np.linspace(xstart, xend, nx));
    yind_block = np.round(np.linspace(ystart, yend, ny));
    zind_block = np.round(np.linspace(zstart, zend, nz));
    
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    
    ind_block = ind_block.astype(int);
    
    return ind_block, ind_brain

def block2brain(blocks, inds):
    vol_brain = np.zeros([145,174,145, blocks.shape[-1]])
    vol_count = np.zeros([145,174,145, blocks.shape[-1]])
    
    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :]
        #print(vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :].shape,blocks[tt, :, :, :, :].shape)
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + blocks[tt, :, :, :, :]
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + 1.
    
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    return vol_brain, vol_count 
def save_nii(fpNii, data, fpRef):
    
    new_header = header=nb.load(fpRef).header.copy()    
    new_img = nb.nifti1.Nifti1Image(data, None, header=new_header)    
    nb.save(new_img, fpNii)  
