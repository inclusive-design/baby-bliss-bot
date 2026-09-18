#!/bin/bash

# Copyright (c) 2025, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=add_single_token_gloss_symbols
#SBATCH --time 1-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
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

module load StdEnv/2023

pip install torch==2.6.0 transformers==4.50.3

pip list

echo "=== Add single-token-gloss symbols into a Llama model with job ID $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/add-single-token-gloss-symbols/add_single_token_gloss_symbols.py ~/bliss_gloss/add-single-token-gloss-symbols/data/bliss_gloss_cleaned.json ~/bliss_gloss/add-single-token-gloss-symbols/output/bliss_ids_added.json ~/bliss_gloss/add-single-token-gloss-symbols/output/bliss_ids_not_added.json
