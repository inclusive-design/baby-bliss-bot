#!/bin/bash

# Copyright (c) 2024, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=llama2-bliss-gloss-similarity-comparison
#SBATCH --time 2-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
 
pip install --upgrade pip
module load python/3.11.5

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip

module load StdEnv/2023
pip install --no-index transformers torch transformers[sentencepiece] sentencepiece

echo "=== Check Bliss gloss similarity from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/bliss_gloss_similarity.py ~/bliss_gloss/bliss_gloss.txt 0.5
