#!/bin/bash

# Copyright (c) 2023, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE
#
# For use with the Pytorch version of StyleGAN2-ADA's generate tool,
# `generate.py`.  This requires that the Pytorch version of StyleGAN2-ADA
# project has been cloned and the environment properly set up.
#

#SBATCH --job-name=StyleGAN-2
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM) -- 1 hour
#SBATCH --account=def-whkchun
#SBATCH --output=%x-%N-%j.out  # %N for node name, %j for jobID

# These modules needed for the def-whkchun cluster for cuda
# If using the ctb-whckchun cluster, use #SBATCH --account=ctb-whckchun above,
# and instead load these modules for cuda:
# module load cuda cudnn
module load nixpkgs/16.09  intel/2018.3  cuda/10.0.130 cudnn/7.5

# Set the virtual environment
source ~/BlissStyleGAN/StyleGAN2/pytorch/bin/activate

# Latest model from training (pickle file)
MODEL_FILE=./pytorch-ada-results/00003-preppedBlissSingleCharGrey-auto1-resumecustom/network-snapshot-000120.pkl

# Create a folder for the training results.
OUTPUT_DIR=~/BlissStyleGAN/StyleGAN2/pytorch-ada-generate
mkdir -p "$OUTPUT_DIR"

# GENERATING
#
# This third command is resuming for another 12 hours, using latest model
python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/generate.py --outdir="$OUTPUT_DIR" --trunc=0.5 --seeds=600-605 --network="$MODEL_FILE"


