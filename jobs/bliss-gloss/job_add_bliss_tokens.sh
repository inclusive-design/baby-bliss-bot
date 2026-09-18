#!/bin/bash

# Copyright (c) 2025, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=llama3-add-bliss-first-single-tokens
#SBATCH --time 2-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
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

echo "=== Add first single Bliss tokens to Llama 3.1 8B model from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/add_bliss_tokens.py ~/bliss_gloss/data/bliss_gloss_cleaned_synonyms.json ~/bliss_gloss/outputs/models/llama-first-single-token-8B ~/bliss_gloss/outputs/bliss_ids_added_new.json ~/bliss_gloss/outputs/bliss_ids_not_added_new.json
