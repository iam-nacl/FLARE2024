import os
import nibabel as nib
import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import dice
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def load_nii(file_path):
    return nib.load(file_path).get_fdata()


def dice_coefficient(pred, true):
    num_classes = 14  
    dice_scores = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id).astype(int)
        true_class = (true == class_id).astype(int)

        intersection = np.sum(pred_class * true_class)
        if (np.sum(pred_class) + np.sum(true_class)) > 0:
            dice_score = 2. * intersection / (np.sum(pred_class) + np.sum(true_class))
            dice_scores.append(dice_score)
        else:
            dice_scores.append(1.0)  

    return np.mean(dice_scores)


def process_case(case_num):
    img_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_FLAREUnLabeledCase2000_blackbean/imagesTr/Case_{case_num:05d}_0000.nii.gz"
    label1_path = f"/data/FLARE24/data/output_training/Task102/epoch1000/Case_{case_num:05d}.nii.gz"
    label2_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_FLAREUnLabeledCase2000_blackbean/labelsTr/Case_{case_num:05d}.nii.gz"
    label3_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task105_FLAREUnLabeledCase2000_aladdin5/labelsTr/Case_{case_num:05d}.nii.gz"

    label1 = load_nii(label1_path)
    label2 = load_nii(label2_path)
    label3 = load_nii(label3_path)

    dice12 = dice_coefficient(label1, label2)
    dice13 = dice_coefficient(label1, label3)

    return case_num, dice12, dice13


def main():
    results = []
    batch_size = 100
    for i in range(1, 2001, batch_size):
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_case, j) for j in range(i, min(i + batch_size, 2001))]
            for future in futures:
                results.append(future.result())
        
        del futures
        import gc
        gc.collect()  

    results_df = pd.DataFrame(results, columns=['Case', 'Dice12', 'Dice13'])
    results_df.to_json('/home/wzh/code/FLARE24/nnunet/hardData_select/dice_results.json', orient='records')
    results_df.to_excel('/home/wzh/code/FLARE24/nnunet/hardData_select/dice_results.xlsx', index=False)

    plt.hist(results_df['Dice12'], bins=50, alpha=0.5, label='Dice12')
    plt.hist(results_df['Dice13'], bins=50, alpha=0.5, label='Dice13')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('/home/wzh/code/FLARE24/nnunet/dice_distribution.png')


if __name__ == "__main__":
    main()
