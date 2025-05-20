# s_DeepDTI_trainCNN.py
#
#   A script for trian the convolutional neural network in DeepDTI.
#
#   Source code:
#       https://github.com/qiyuantian/DeepDTI/blob/main/s_DeepDTI_trainCNN.py
#
#   Reference:
#       [1] Tian Q, Bilgic B, Fan Q, Liao C, Ngamsombat C, Hu Y, Witzel T,
#       Setsompop K, Polimeni JR, Huang SY. DeepDTI: High-fidelity
#       six-direction diffusion tensor imaging using deep learning.
#       NeuroImage. 2020;219:117017. 
#
#       [2] Tian Q, Li Z, Fan Q, Ngamsombat C, Hu Y, Liao C, Wang F,
#       Setsompop K, Polimeni JR, Bilgic B, Huang SY. SRDTI: Deep
#       learning-based super-resolution for diffusion tensor MRI. arXiv
#       preprint. 2021; arXiv:2102.09069.
#
# (c) Qiyuan Tian, Harvard, 2021

# %% load modual

import os
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# %% load DnCNN model and utility library

import qtlib as qtlib
from model.KAN_noddi import *
from torch.utils.data import DataLoader
from dataloader_noddi import PartANODDIFT
dataset = PartANODDIFT(test = 1)
batch_size = 1
shuffle = False
num_workers = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
print(len(dataloader))
# model = KAN_Convolution_Network().to('cuda:1')
model = UKAN_8().to('cuda:1')
model.eval()
# model.load_state_dict(torch.load('/Data/Users/cyf/DWI/noddi-DeepDTI/model_weights_1_final.pth'))
model.load_state_dict(torch.load('/Data/Users/cyf/DWI/noddi-DeepDTI/model_weights_2_final.pth'))
def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated(torch.device('cuda:1')) / 1024**2  # 转换为 MB
    cached = torch.cuda.memory_cached(torch.device('cuda:1')) / 1024**2  # 转换为 MB
    print(f"Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
with torch.no_grad():
    i = 0
    
    outs = None
    inds = None
    labels = None
    for batch in dataloader:
        i += 1
        img, mask, label, ind, subj = batch
        #print("ind",ind.shape)
        img = img.permute(0,4,1,2,3).to('cuda:1')
        mask = mask.permute(0,4,1,2,3).to('cuda:1')
        label = label.permute(0,4,1,2,3).to('cuda:1')
        print(img.shape, mask.shape,label.shape)
        out = (model(img*mask)*mask).permute(0,2,3,4,1)
        label = label.permute(0,2,3,4,1)
        #print("loss",loss,((label[:,:,:,:,0]-out[:,:,:,:,0])**2).mean(),((label[:,:,:,:,1]-out[:,:,:,:,1])**2).mean() , ((label[:,:,:,:,2]-out[:,:,:,:,2])**2).mean())
        if outs == None:
            outs = out
            inds = ind
            labels = label
        else:
            outs = torch.cat((outs, out), dim=0)
            inds = torch.cat((inds, ind), dim=0)
            labels = torch.cat((labels, label), dim=0)
        print(subj)
        if i == 12:
            i = 0

            #torch.Size([12, 6]) torch.Size([12, 64, 64, 64, 4])
            #print("before block",inds.shape,outs.shape)
            outs = outs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            print(outs.shape,labels.shape)
            #(36, 64, 64, 64, 7) (36, 6) (145, 174, 145, 1)
            image,_ = qtlib.block2brain(outs,inds)
            gt,_ = qtlib.block2brain(labels,inds)
            print("after block",image.shape)
            qtlib.save_nii(f"/Data/Users/cyf/DWI/noddi-DeepDTI/result/{subj[0]}_fiso_debug.nii.gz",image[:,:,:,0]/1000,f"/Data/shared_data/HCP_MWU/{subj[0]}/diff-Gt-matlab/{subj[0]}noddiparameters_fiso.nii")
            qtlib.save_nii(f"/Data/Users/cyf/DWI/noddi-DeepDTI/result/{subj[0]}_ficvf_debug.nii.gz",image[:,:,:,1]/1000,f"/Data/shared_data/HCP_MWU/{subj[0]}/diff-Gt-matlab/{subj[0]}noddiparameters_ficvf.nii")
            qtlib.save_nii(f"/Data/Users/cyf/DWI/noddi-DeepDTI/result/{subj[0]}_odi_debug.nii.gz",image[:,:,:,2]/1000,f"/Data/shared_data/HCP_MWU/{subj[0]}/diff-Gt-matlab/{subj[0]}noddiparameters_odi.nii")
            qtlib.save_nii(f"/Data/Users/cyf/DWI/noddi-DeepDTI/result/{subj[0]}_gt_fiso.nii.gz",gt[:,:,:,0],f"/Data/shared_data/HCP_MWU/{subj[0]}/diff-Gt-matlab/{subj[0]}noddiparameters_fiso.nii")
            qtlib.save_nii(f"/Data/Users/cyf/DWI/noddi-DeepDTI/result/{subj[0]}_gt_ficvf.nii.gz",gt[:,:,:,1],f"/Data/shared_data/HCP_MWU/{subj[0]}/diff-Gt-matlab/{subj[0]}noddiparameters_ficvf.nii")
            qtlib.save_nii(f"/Data/Users/cyf/DWI/noddi-DeepDTI/result/{subj[0]}_gt_odi.nii.gz",gt[:,:,:,2],f"/Data/shared_data/HCP_MWU/{subj[0]}/diff-Gt-matlab/{subj[0]}noddiparameters_odi.nii")
            outs = None
            inds = None
