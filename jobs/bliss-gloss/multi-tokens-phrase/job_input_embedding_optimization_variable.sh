#!/bin/bash

# Copyright (c) 2025, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=input_embedding_optimization_variable
#SBATCH --time 1-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
 
pip install --upgrade pip
module load python/3.13

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip

module load StdEnv/2023 rust/1.85.0 arrow/19.0.1 gcc/13.3

pip install torch==2.6.0 transformers==4.50.3

pip list

echo "=== Optimize input embedding for a multiple tokens phrase Llama with job ID $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/multi-tokens-phrase/input_embedding_optimization_variable.py > ~/bliss_gloss/multi-tokens-phrase/logs/input_embedding_optimization_variable.log
