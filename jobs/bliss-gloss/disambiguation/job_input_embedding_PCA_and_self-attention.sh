#!/bin/bash

# Copyright (c) 2025, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=input_embedding_PC_and_self-attention
#SBATCH --time 1-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
#SBATCH --mail-user=cli@ocadu.ca
#SBATCH --mail-type=START,END,FAIL

pip install --upgrade pip
module load python/3.13

source ~/.virtualenvs/transformers_torch/bin/activate
pip list

echo "=== Disambiguate synonyms using PCA and self-attention methods with job ID $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/disambiguation/input_embedding_PCA_and_self-attention.py > ~/bliss_gloss/disambiguation/test_results/IE_average_PCs_OE_calculated_24852_break_fracture_injury_damage.log
