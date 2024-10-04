import nibabel as nib
import numpy as np
import os

# 标签范围定义
VALID_LABELS = set(range(14))  # 有效标签从 0 到 13

# 文件夹路径
folders = [
    '/data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task105_FLAREUnLabeledCase2000_aladdin5/labelsTr/'
]


def check_and_fix_labels(folder):
    print(f"Checking folder: {folder}")
    for filename in os.listdir(folder):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(folder, filename)
            img = nib.load(file_path)
            data = img.get_fdata()

            # 获取不符合标签范围的索引
            invalid_indices = np.where(~np.isin(data, list(VALID_LABELS)))
            if len(invalid_indices[0]) > 0:
                print(f"Found invalid labels in {filename}. Fixing...")

                # 将无效标签修改为 0
                data[invalid_indices] = 0

                # 保存修改后的数据
                new_img = nib.Nifti1Image(data, img.affine, img.header)
                nib.save(new_img, file_path)

    print(f"Finished checking folder: {folder}")


# 按顺序检查每个文件夹
for folder in folders:
    check_and_fix_labels(folder)
