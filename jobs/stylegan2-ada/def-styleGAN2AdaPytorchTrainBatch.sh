#!/bin/bash

# Copyright (c) 2023, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE
#
# For use with the Pytorch version of StyleGAN2-ADA's training script,
# `train.py`.  This requires that the Pytorch version of StyleGAN2-ADA
# project has been cloned and the environment properly set up.
#
# Inputs: preppedBlissSinceCharGrey.tar - a tarfile of .tfrecord` files.  The
#         images with are 256x256
# Ouputs: pytorch-ada-results - a folder for saving the model and sample results

#SBATCH --job-name=StyleGAN-2
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-24:00      # time (DD-HH:MM) -- 24 hours
#SBATCH --account=def-whkchun
#SBATCH --output=%x-%N-%j.out  # %N for node name, %j for jobID

# These modules needed for the def-whkchun cluster
# If using the ctb-whckchun cluster, use #SBATCH --account=ctb-whckchun above,
# and instead load these modules for cuda:
# module load cuda cudnn
module load nixpkgs/16.09  intel/2018.3  cuda/10.0.130 cudnn/7.5

# Set the virtual environment
source ~/BlissStyleGAN/StyleGAN2/pytorch/bin/activate

# Copy the (prepared) data to the SLURM_TMPDIR
echo -n "SLURM temporary directory: "
echo "$SLURM_TMPDIR"
echo
DATA_DIR="$SLURM_TMPDIR/preppedBlissSingleCharGrey"
tar xf ~/BlissStyleGAN/StyleGAN2/preppedBliss4Pytorch.tar -C "$SLURM_TMPDIR"
ls -R "$DATA_DIR"

# Create a folder for the training results.
OUTPUT_DIR=./pytorch-ada-results
mkdir -p "$OUTPUT_DIR"

# TRAINING
#
# This first command is for starting from scratch -- no resume argument.
python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/train.py --outdir="$OUTPUT_DIR" --data="$DATA_DIR" --snap=10

# This second command was after training for 24 hours and resuming for another
# 4.5 hours, picking up from the model generated after 24 hours.  The actual
# arguments may differ for another instnace of a first run
# python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/train.py --outdir="$OUTPUT_DIR" --data="$DATA_DIR" --snap=10 --resume="$OUTPUT_DIR/00000-preppedBlissSingleCharGrey-auto1/network-snapshot-000880.pkl"

# This third command is resuming for another 15 hours, using latest model.
# Again, the actual values here may differ for different groups of runs.
# python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/train.py --outdir="$OUTPUT_DIR" --data="$DATA_DIR" --snap=10 --resume="$OUTPUT_DIR/00001-preppedBlissSingleCharGrey-auto1-resumecustom/network-snapshot-000440.pkl"

