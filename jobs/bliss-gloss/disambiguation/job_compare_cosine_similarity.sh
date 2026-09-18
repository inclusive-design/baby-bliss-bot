#!/bin/bash

# Copyright (c) 2025, Inclusive Design Institute
#
# Licensed under the BSD 3-Clause License. You may not use this file except
# in compliance with this License.
#
# You may obtain a copy of the BSD 3-Clause License at
# https://github.com/inclusive-design/baby-bliss-bot/blob/main/LICENSE

#SBATCH --job-name=compare_cosine_similarity
#SBATCH --time 1-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
#SBATCH --mail-user=cli@ocadu.ca
#SBATCH --mail-type=START,END,FAIL

pip install --upgrade pip
module load python/3.13

source ~/.virtualenvs/transformers_torch/bin/activate
pip list

echo "=== Compare cosine similarity with job ID $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/bliss_gloss/disambiguation/compare_cosine_similarity.py ~/bliss_gloss/disambiguation/test_results/compare_input_embeddings_cosine_similarity_result.json > ~/bliss_gloss/disambiguation/test_results/compare_input_embeddings_cosine_similarity_result.log
