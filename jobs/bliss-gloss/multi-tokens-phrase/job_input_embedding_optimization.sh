#!/bin/bash

# Copyright (c) 2025, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=llama3-optimize-input-embedding
#SBATCH --time 4-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
 
pip install --upgrade pip
module load python/3.12

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip

pip install transformers torch

echo "=== Optimize input embedding for multi-token phrase with job ID $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/multi-tokens-phrase/input_embedding_optimization.py 0.005 1000 100 > ~/bliss_gloss/multi-tokens-phrase/test_results/result_of_optimize_input_embedding.log
