import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

def main():
    threshold = 0.90
    results_df = pd.read_json('/home/wzh/code/FLARE24/nnunet/hardData_select/dice_results.json')

    
    easy_samples = results_df[(results_df['Dice12'] > threshold) & (results_df['Dice13'] > threshold)]

    

    
    for _, row in easy_samples.iterrows():
        case_num = int(row['Case'])
        img_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_FLAREUnLabeledCase2000_blackbean/imagesTr/Case_{case_num:05d}_0000.nii.gz"
        label_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_FLAREUnLabeledCase2000_blackbean/labelsTr/Case_{case_num:05d}.nii.gz"

        dest_img_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task113_FLARE_EasySample0.90/imagesTr/Case_{case_num:05d}_0000.nii.gz"
        dest_label_path = f"/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task113_FLARE_EasySample0.90/labelsTr/Case_{case_num:05d}.nii.gz"

        os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)

        shutil.copy(img_path, dest_img_path)
        shutil.copy(label_path, dest_label_path)

if __name__ == "__main__":
    main()