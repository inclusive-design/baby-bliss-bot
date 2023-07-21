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
# Inputs: network-snapshot-nnnnnn.pkl - a StyleGAN2-ADA model file where the
#         six 'nnnnnn' digits are replaced with the actual digits used.
# Ouputs: pytorch-ada-generate - a folder for saving the generated images.

#SBATCH --job-name=StyleGAN-2
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM) -- 1 hour
#SBATCH --account=ctb-whkchun
#SBATCH --output=%x-%N-%j.out  # %N for node name, %j for jobID

# These modules needed for the ctb-whkchun cluster
module load cuda cudnn

# Set the virtual environment
source ~/BlissStyleGAN/StyleGAN2/pytorch/bin/activate

# Latest model from training (pickle file)
MODEL_FILE=./pytorch-ada-results/00002-preppedBlissSingleCharGrey-auto1-resumecustom/network-snapshot-001640.pkl

# Create a folder for the training results.
OUTPUT_DIR=~/BlissStyleGAN/StyleGAN2/pytorch-ada-generate
mkdir -p "$OUTPUT_DIR"

# Generate...
python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/generate.py --outdir="$OUTPUT_DIR" --trunc=0.5 --seeds=200,330,400 --network="$MODEL_FILE"


