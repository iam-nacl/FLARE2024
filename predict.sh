# !/bin/bash -e
export RESULTS_FOLDER="$PWD/nnUNet_trained_models/"
export nnUNet_raw_data_base="$PWD/nnUNet_raw_data_base/"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed/"

CUDA_VISIBLE_DEVICES="" nnUNet_predict -i .//inputs/  -o ./outputs  -t 114  -p nnUNetPlansLightUnet -modelsavename LightUnet3_Task114_epoch1500  -m 3d_fullres \
 -tr nnUNetTrainerV2_LightUnet  -f all  --mode fastest --disable_tta --all_in_gpu False

