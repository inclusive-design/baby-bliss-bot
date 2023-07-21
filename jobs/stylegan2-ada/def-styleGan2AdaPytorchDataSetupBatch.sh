#!/bin/bash

# Copyright (c) 2023, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE
#
# For use with the Pytorch version of StyleGAN2-ADA's data preperation tool,
# `dataset_tool.py`.  This requires that the Pytorch version of StyleGAN2-ADA
# project has been cloned and the environment properly set up.
#
# Inputs: preppedBlissSinceCharGrey.tar - a tarfile of `.png` files.  The
#         images are greyscae with dimensions 256x256 pixels.
# Ouputs: preppedBliss4Pytorch.tar - a tarfile of the prepared images, stored
#                                    in the home directory.

#SBATCH --job-name=StyleGAN-2
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM) -- 1 hour
#SBATCH --account=def-whkchun
#SBATCH --output=%x-%N-%j.out  # %N for node name, %j for jobID

# These modules needed for the def-whkchun cluster.
# If using the ctb-whckchun cluster, use #SBATCH --account=ctb-whckchun above,
# and instead load these modules for cuda:
# module load cuda cudnn
module load nixpkgs/16.09  intel/2018.3  cuda/10.0.130 cudnn/7.5

# Set the virtual environment
source ~/BlissStyleGAN/StyleGAN2/pytorch/bin/activate

# Copy the unprepared data to the SLURM_TMPDIR
echo -n "SLURM temporary directory: "
echo "$SLURM_TMPDIR"
echo
mkdir $SLURM_TMPDIR/blissSingleCharGrey
tar xf ~/BlissStyleGAN/blissSingleCharsGrey.tar -C "$SLURM_TMPDIR/blissSingleCharGrey"
ls -R $SLURM_TMPDIR/blissSingleCharGrey

# Prepare the dataset
PREPPED_DATA_DIR="preppedBlissSingleCharGrey"
OUTPUT_DIR="$SLURM_TMPDIR/$PREPPED_DATA_DIR"
mkdir "$OUTPUT_DIR"
INPUT_DIR="$SLURM_TMPDIR/blissSingleCharGrey/blissSingleCharsInGrayscale"
python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/dataset_tool.py --source "$INPUT_DIR" --dest "$OUTPUT_DIR"
STATUS=$?
echo "Data prep exit status is $STATUS"
ls -R "$OUTPUT_DIR"

# Tar the prepped data and copy it back to "home"
if [ $STATUS == 0 ]; then
    tar cf ~/BlissStyleGAN/StyleGAN2/preppedBliss4Pytorch.tar -C "$SLURM_TMPDIR" "$PREPPED_DATA_DIR"
else
    echo "dataset_tool.py failed with exit status $STATUS"
fi
echo Done!






