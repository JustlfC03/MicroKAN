# -*- coding: utf-8 -*-

"""
@Time    : 2022/9/26 14:30
@Author  : Zihan Li
@Email   : zihan-li18@outlook.com
"""

import numpy as np
import nibabel as nb
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit
import qtlib
import warnings
import os
import glob
warnings.filterwarnings("ignore")
def EvalDtiNew(fpCal,fpRef,tmask,bmask,subj,metrics=["fiso","ficvf","odi"]):
    error = {}
    for ii in range(0,len(metrics)):
        ref = nb.load(os.path.join(fpRef,subj,"diff-Gt-matlab",subj +"noddiparameters_"+ metrics[ii] +".nii")).get_fdata()
        cal = nb.load(os.path.join(fpCal,subj +"_"+ metrics[ii] +"_debug.nii.gz")).get_fdata()
        # cal = nb.load(os.path.join("/Data/Users/DiffusionGroup/Reference/HCP",subj,"diff-NODDI3151515-matlab","example_"+metrics[ii] +".nii")).get_fdata()
        # print("ref.shape",ref.shape)
        # print("cal.shape",cal.shape)
        if metrics[ii] in ["fiso"]:
            isovf = np.nanmean(abs(ref-cal)[tmask>0])
            error[metrics[ii]] = isovf
        else:
            bmask = bmask[isovf<0.8]
            mask = bmask*tmask
            mask = mask.squeeze()
            mae = np.nanmean(abs(ref-cal)[mask>0])
            error[metrics[ii]] = mae
    return error
if __name__ == "__main__":
    for subj in ['mwu100307', 'mwu114419', 'mwu101107', 'mwu113619', 'mwu103818', 'mwu105115', 'mwu106016', 'mwu110411', 'mwu115320', 'mwu111716']:
    #for subj in ['mwu103818']:
        mask = nb.load(f"/Data/shared_data/HCP_MWU/{subj}/diff/{subj}_diff_mask.nii.gz").get_fdata()
        bmask = mask
        result_dict = EvalDtiNew("/Data/Users/cyf/DWI/noddi-DeepDTI/result","/Data/shared_data/HCP_MWU/",mask,bmask,subj)
        print(subj,result_dict)