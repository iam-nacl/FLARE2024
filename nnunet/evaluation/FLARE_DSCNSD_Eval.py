# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:59:48 2022

@author: 12593
"""
import sys
import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
join = os.path.join
basename = os.path.basename
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument(
#     'fold',
#     type=str,
# )

args = parser.parse_args()

gt_path = ""
seg_path = ""
save_path = ""



filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

seg_metrics_DSC = OrderedDict()
seg_metrics_DSC['Name'] = list()
seg_metrics_NSD = OrderedDict()
seg_metrics_NSD['Name'] = list()
# label_tolerance = OrderedDict({'spleen':3, 'right kidney':3, 'left kidney':3, 'gallbladder':2, 
#                    'esophagus':3, 'liver': 5, 'stomach': 5, 'aorta': 2, 'inferior vena cava': 2,
#                    'portal vein and splenic vein':2, 'pancreas': 5, 'right adrenal gland': 2, 'left adrenal gland':2})
label_tolerance = OrderedDict({'spleen':2, 'right kidney':2, 'left kidney':2, 'gallbladder':2, 
                   'esophagus':2, 'liver': 2, 'stomach': 2, 'aorta': 2, 'inferior vena cava': 2,
                   'portal vein and splenic vein':2, 'pancreas': 2, 'right adrenal gland': 2, 'left adrenal gland':2})

for organ in label_tolerance.keys():
    seg_metrics_DSC['{}_DSC'.format(organ)] = list()
    seg_metrics_NSD['{}_NSD'.format(organ)] = list()

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper



for name in tqdm(filenames):
    seg_metrics_DSC['Name'].append(name)
    seg_metrics_NSD['Name'].append(name)
    # load grond truth and segmentation
    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

    for i, organ in enumerate(label_tolerance.keys(),1):
        if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
            DSC_i = 0
            NSD_i = 0
        else:
            if i==5 or i==8 or i==9: # for Esophagus, Aorta, and IVC only evaluate the labelled slices in ground truth
                z_lower, z_upper = find_lower_upper_zbound(gt_data==i)
                organ_i_gt, organ_i_seg = gt_data[:,:,z_lower:z_upper]==i, seg_data[:,:,z_lower:z_upper]==i
            else:
                organ_i_gt, organ_i_seg = gt_data==i, seg_data==i
            
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
        
        seg_metrics_DSC['{}_DSC'.format(organ)].append(round(DSC_i, 4))
        seg_metrics_NSD['{}_NSD'.format(organ)].append(round(NSD_i, 4))

dataframe_DSC = pd.DataFrame(seg_metrics_DSC)
dataframe_DSC.to_csv(save_path + 'FLARE_DSC.csv', index=False)
dataframe_NSD = pd.DataFrame(seg_metrics_NSD)
dataframe_NSD.to_csv(save_path + 'FLARE_NSD.csv', index=False)

case_avg_DSC = dataframe_DSC.mean(axis=0, numeric_only=True)
case_avg_NSD = dataframe_NSD.mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Average DSC for {basename(seg_path)}: {case_avg_DSC.mean()}')
print(f'Average NSD for {basename(seg_path)}: {case_avg_NSD.mean()}')

print('DSC organ:', case_avg_DSC)
print('NSD organ:', case_avg_NSD)
# organs8_DSC = [x for inx, x in enumerate(case_avg_DSC) if inx in [0,1,2,3,5,6,7,10]]
# print('8 selected organ:', organs8_DSC)
# print('Average DSC for 8 selected organ:', sum(organs8_DSC)/len(organs8_DSC))
# organs8_NSD = [x for inx, x in enumerate(case_avg_NSD) if inx in [0,1,2,3,5,6,7,10]]
# print('8 selected organ:', organs8_NSD)
# print('Average NSD for 8 selected organ:', sum(organs8_NSD)/len(organs8_NSD))


print(20 * '<')