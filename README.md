
# CDL-UNet: Curriculum-Driven Lightweight 3D U-Net for Abdominal Organ Segmentation


This repository is the official implementation of [CDL-UNet: Curriculum-Driven Lightweight 3D U-Net for Abdominal Organ Segmentation](TBA).


## Environments and Requirements

- **Operating System**: Ubuntu 20.04.4 LTS
- **Programming Language**: Python 3.9
- **Deep Learning Framework**: Torch 2.3.1
- **CUDA Version**: 12.1

To install the requirements, use the following command:

```
pip install -r requirements.txt
```

You can reproduce our method step by step as outlined below.

## Dataset

- Dataset is from [https://www.codabench.org/competitions/2320/](https://www.codabench.org/competitions/2320/).
- Prepare the data by placing different parts of the original dataset in the `nnUNet_raw_data_base` folder with the following names:
    - Task102_FLARELabeledCase
    - Task103_FLAREUnLabeledCase2000_blackbean
    - Task105_FLAREUnLabeledCase2000_aladdin5

## Preprocessing

We first clean noisy data in some samples:

```
python preprocessing/checkBadData.py
```

Next, we use the nnUNet preprocessing method:

```
nnUNet_plan_and_preprocess -t 102 -pl3d ExperimentPlanner3D_BigUnet -pl2d None
nnUNet_plan_and_preprocess -t 103 -pl3d ExperimentPlanner3D_LightUnet -pl2d None
nnUNet_plan_and_preprocess -t 105 -pl3d ExperimentPlanner3D_LightUnet -pl2d None
```
By running the above commands, you complete the basic data preprocessing step.

## Training

1. First, we use a larger nnU-Net model to generate additional reference labels for 2000 unlabeled images and save the predicted labels:

```
nnUNet_train 3d nnUNetTrainerV2_BigUnet 102 all -p nnUNetPlansFLAREBigUnet
nnUNet_predict -i /data/FLARE24/data/nnUNet_raw_data_base/nnUNet_raw_data/Task105_FLAREUnLabeledCase2000_aladdin5/imagesTr/ -o <path_to_output_data> -t 102 -p nnUNetPlansBigUnet -m 3d_fullres -tr nnUNetTrainerV2_BigUnet -f all --mode fastest --disable_tta --modelsavename BigUnet3d_epoch1000
```

2. Next, we build the Data-level Curriculum Learning dataset. First, calculate the Reliability Score for each label:

```
python hardData_select/ReliabilityScore_cul.py
```

Then, based on the Reliability Score, select labels to build new datasets `Task113_FLARE_EasySample0.90` and `Task114_FLARE_HardSample0.90`:

```
python hardData_select/ReasyData_sel.py
python hardData_select/RhardData_sel.py
```

Finally, use the nnUNet preprocessing method to process the Data-level Curriculum Learning datasets `Task113_FLARE_EasySample0.90` and `Task114_FLARE_CurriculumSample0.90`:

```
nnUNet_plan_and_preprocess -t 113 -pl3d ExperimentPlanner3D_LightUnet -pl2d None
nnUNet_plan_and_preprocess -t 114 -pl3d ExperimentPlanner3D_LightUnet -pl2d None
```

3. Data-level Curriculum Learning Training:

- Easy Course:

```
nnUNet_train 3d_fullres nnUNetTrainerV2_LightUnet 113 all -p nnUNetPlansLightUnet -modelsavename LightUnet3_Task113_epoch1000
```

- Hard Course:

```
nnUNet_train 3d_fullres nnUNetTrainerV2_LightUnet 114 all -p nnUNetPlansLightUnet -modelsavename LightUnet3_Task114_epoch1500 -pretrained_weights /data/FLARE24/data/nnUNet_trained_models/nnUNet/3d_fullres/Task113_FLARE_EasySample0.90/LightUnet3_Task114_epoch1000
```

## Trained Models

You can download the trained models here:

- [My awesome model](https://github.com/iam-nacl/FLARE2024/tree/main/nnUNet_trained_models/nnUNet/3d_fullres/Task114_FLARE_CurriculumSample0.9/LightUnet3_Task114_epoch1500/nnUNetTrainerV2_LightUnet__nnUNetPlansLightUnet/all) trained on the dataset above.

## Inference

To infer the testing cases, run this command:

```
nnUNet_predict -i <path_to_data> -o <path_to_output_data> -t 114 -p nnUNetPlansLightUnet -m 3d_fullres -tr nnUNetTrainerV2_LightUnet -f all --mode fastest --disable_tta --modelsavename <trained_model_name>
```

## Evaluation

To compute evaluation metrics, run the following command:

```
python evaluation/FLARE_DSCNSD_Eval.py
```

Make sure to update the necessary paths in the script.

## Results

Our method achieves the following performance on [MICCAI FLARE 2024 Task2: Abdominal CT Organ Segmentation on Laptop](https://www.codabench.org/competitions/2320/):

| Target                | DSC (%) Public Validation    | NSD (%) Public Validation    | DSC (%) Online Validation   | NSD (%) Online Validation   |
|-----------------------|------------------------------|------------------------------|-----------------------------|-----------------------------|
| Liver                 | 97.10 ± 1.36                 | 91.91 ± 5.42                 | 96.92                       | 97.32                       |
| Right Kidney          | 94.51 ± 13.70                | 93.84 ± 14.63                | 92.19                       | 93.84                       |
| Spleen                | 96.39 ± 1.80                 | 95.80 ± 5.18                 | 94.06                       | 95.16                       |
| Pancreas              | 89.99 ± 2.37                 | 90.01 ± 5.12                 | 85.95                       | 96.08                       |
| Aorta                 | 95.30 ± 1.90                 | 98.49 ± 3.32                 | 95.54                       | 98.51                       |
| Inferior vena cava     | 91.91 ± 2.94                 | 93.07 ± 4.61                 | 90.75                       | 92.93                       |
| Right adrenal gland   | 82.25 ± 7.54                 | 95.93 ± 5.54                 | 83.41                       | 95.92                       |
| Left adrenal gland    | 83.17 ± 11.19                | 95.63 ± 9.64                 | 82.36                       | 94.49                       |
| Gallbladder           | 80.75 ± 30.51                | 83.14 ± 31.45                | 79.25                       | 80.38                       |
| Esophagus             | 85.17 ± 5.61                 | 90.71 ± 7.61                 | 81.34                       | 92.22                       |
| Stomach               | 93.34 ± 3.19                 | 88.62 ± 7.41                 | 92.13                       | 94.76                       |
| Duodenum              | 86.92 ± 5.14                 | 87.55 ± 6.09                 | 80.12                       | 92.88                       |
| Left kidney           | 93.34 ± 14.75                | 92.69 ± 16.08                | 93.61                       | 94.95                       |
| **Average**           | **90.01 ± 7.19**             | **92.11 ± 9.39**             | **88.28**                   | **93.80**                   |


## Contributing

If you want to contribute, please follow the usual GitHub workflow (fork, pull request) and feel free to submit bug reports, feature requests, and suggestions.

## Acknowledgement



We thank the contributors of the public datasets used in this project. Additionally, we acknowledge that our code architecture is based on the work of MIC-DKFZ/nnUNet, and some files drew inspiration from Ziyan-Huang/FLARE22. 








