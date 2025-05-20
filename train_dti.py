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
from model.KAN_dti import *
from torch.utils.data import DataLoader
from dataloader_dti import PartANODDIFT
dataset = PartANODDIFT(test = 0)
batch_size = 1
shuffle = False
num_workers = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
print(len(dataloader))
# model = KAN_Convolution_Network().to('cuda:1')
model = UKAN_8().to('cuda:1')
model.train()
# model.load_state_dict(torch.load('/Data/Users/cyf/DWI/DeepDTI-main/model_weights_final.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
epoch = 100
for i in range(epoch):
    epoch_loss = 0
    for batch in dataloader:
        imgs, mask, labels = batch
        imgs = imgs.permute(0,4,1,2,3).to('cuda:1')
        mask = mask.permute(0,4,1,2,3).to('cuda:1')
        labels = labels.permute(0,4,1,2,3).to('cuda:1')
        # 在这里进行模型训练或评估
        #print(imgs.shape, mask.shape,labels.shape)
        out = model(imgs*mask)*mask
        labels[:,1:] = labels[:,1:]*1000
        loss = ((labels[:,0]-out[:,0])**2 + (labels[:,1]-out[:,1])**2 + (labels[:,2]-out[:,2])**2 + (labels[:,3]-out[:,3])**2).mean()
        #print("loss",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    print("epoch_loss",epoch_loss)
    if i % 10 == 0:
        torch.save(model.state_dict(), f'/Data/Users/cyf/DWI/DeepDTI-main/model_weights_1_{i}.pth')

# torch.save(model.state_dict(), '/Data/Users/cyf/DWI/DeepDTI-main/model_weights_2_final.pth')
torch.save(model.state_dict(), '/Data/Users/cyf/DWI/DeepDTI-main/model_weights_1_final.pth')
