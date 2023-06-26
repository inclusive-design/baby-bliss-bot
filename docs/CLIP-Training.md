# Train CLIP

This article documents training a CLIP model with Bliss symbols image files and an annotated json file containing metadata for each image.

The following steps were followed for this training:

## Set Environment

```
# create new env clip_train
conda create -n clip_train python=3.8.5

# activate clip_train
conda activate clip_train

# install pytorch, torchvision
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

# Added additional depedency
pip install future

# install other dependencies
pip install -r requirements.txt
```

## Clone the repoository [clip-training](https://github.com/revantteotia/clip-training)

This repository contains code to train [CLIP](https://github.com/openai/CLIP) on [MS-COCO](https://cocodataset.org/#home) captions.

## Extract Bliss dataset in the directory data

The structure of the COCO dataset was used to prepare the Bliss Annotated dataset. Bliss dataset including images and annotations can be downloaded [here](https://drive.google.com/file/d/1kSE4egEvg2g5wKZLHCFTE1ZijUf0ZC2_/view?usp=sharing)

## Update [./dataloader/data_config.yaml](./dataloader/data_config.yaml)

```
train_img_dir : 'data/bliss/train'
train_annotation_file : 'data/bliss/annotations/bliss_data_annotated_CLIP.json'
```

## Run train.

Take dataset paths from 'dataloader/data_config.yaml'

```
$ python train.py
```

# Results

The results of the training can be downloaded [here - checkpoint_34_3395.pt.tar.gz](https://drive.google.com/file/d/1J_U2yW9MmRa4f23044brM_Winku507ZL/view?usp=sharing)
