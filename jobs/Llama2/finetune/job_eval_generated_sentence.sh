#!/bin/bash
#SBATCH --job-name=llama2-finetune-7b-hf
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

module load StdEnv/2023 gcc/12.3
pip install --no-index spacy sentence_transformers sklearn numpy
pip install textstat
python -m spacy download en_core_web_sm

echo "=== Evaluate generated sentences from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python /home/cindyli/llama2/finetune/eval_generated_sentence.py
