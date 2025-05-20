from torch.utils.data import Dataset
import os
import nibabel as nb
import glob
import torch
import pandas as pd
import numpy as np
import qtlib
import os
import glob
import numpy as np
import nibabel as nb
from torch.utils.data import Dataset

class PartANODDIFT(Dataset):
    def __init__(self, test=0):
        super().__init__()
        self.test = test
        base_path = "/Data/shared_data/HCP_MWU"
        # 使用 glob 模块获取所有以 'mwu' 开头的文件夹
        folder_pattern = os.path.join(base_path, 'mwu*')
        folders = glob.glob(folder_pattern)
        # 获取文件夹名称列表
        all_subj = [os.path.basename(folder) for folder in folders]
        self.test_subj = ['mwu100307', 'mwu114419', 'mwu101107', 'mwu113619', 'mwu103818', 'mwu105115', 'mwu106016', 'mwu110411', 'mwu115320', 'mwu111716']
        self.train_subj = [item for item in all_subj if item not in self.test_subj]
        self.train_len = len(self.train_subj)
        self.test_len = len(self.test_subj)
        
        # 预加载数据
        self.img_blocks = []
        self.mask_blocks = []
        self.gt_blocks = []
        self.inds = []
        self.subjs = []
        if not self.test:
            for subj in self.train_subj:
                self.load_data(subj, base_path)
        else:
            for subj in self.test_subj:
                self.load_data(subj, base_path)
        
        self.total_blocks = len(self.img_blocks)

    def load_data(self, subj, base_path):
        nb0 = 1
        nb1000 = 6
        nb2000 = 0
        nb3000 = 0
        fpsub = os.path.join(base_path, f"{subj}")
        fpImg = os.path.join(fpsub, "diff", f"{subj}_diff.nii.gz")
        fpmask = os.path.join(fpsub, "diff", f"{subj}_diff_mask.nii.gz")
        fpFA = os.path.join(fpsub, "dti", f"{subj}_dti_FA.nii.gz")
        fpMD = os.path.join(fpsub, "dti", f"{subj}_dti_MD.nii.gz")
        fpL1 = os.path.join(fpsub, "dti", f"{subj}_dti_L1.nii.gz")
        fpRD = os.path.join(fpsub, "dti", f"{subj}_dti_RD.nii.gz")
        bval_in = np.loadtxt(glob.glob(os.path.join(fpsub, 'diff', "*_diff.bval"))[0])
        b0index = np.where(bval_in < 100)[0][:nb0]
        b1000index = np.where(abs(bval_in - 1000) < 100)[0][:nb1000]
        b2000index = np.where(abs(bval_in - 2000) < 100)[0][:nb2000]
        b3000index = np.where(abs(bval_in - 3000) < 100)[0][:nb3000]
        selected_indices = np.concatenate([b0index, b1000index, b2000index, b3000index], axis=0)
        
        img = nb.load(fpImg).get_fdata()
        img = img[:, :, :, selected_indices]
        mask = np.expand_dims(nb.load(fpmask).get_fdata(), axis=3)
        FA = nb.load(fpFA).get_fdata()
        MD = nb.load(fpMD).get_fdata()
        L1 = nb.load(fpL1).get_fdata()
        RD = nb.load(fpRD).get_fdata()
        GT = np.stack([FA,MD,L1,RD], axis=3)
        
        ind_block, ind_brain = qtlib.block_ind(mask, sz_block=64, sz_pad=0)
        img_block = qtlib.extract_block(img, ind_block)
        mask_block = qtlib.extract_block(mask, ind_block)
        gt_block = qtlib.extract_block(GT, ind_block)
        for i in range(len(img_block)):
            self.inds.append(ind_block[i])
            self.img_blocks.append(img_block[i])
            self.mask_blocks.append(mask_block[i])
            self.gt_blocks.append(gt_block[i])
            self.subjs.append(subj)

    def __len__(self):
        return self.total_blocks

    def __getitem__(self, idx):
        img_block = torch.tensor(self.img_blocks[idx]).to(torch.float32)
        mask_block = torch.tensor(self.mask_blocks[idx]).to(torch.float32)
        gt_block = torch.tensor(self.gt_blocks[idx]).to(torch.float32)
        ind_block = self.inds[idx]
        subj = self.subjs[idx]
        if self.test == 1:
            return img_block, mask_block, gt_block, ind_block, subj
        else:
            return img_block, mask_block, gt_block
