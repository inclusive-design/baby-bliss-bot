#!/bin/bash
#SBATCH --job-name=llama2-orig-use-7b-hf
#SBATCH --time 10-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j

pip install --upgrade pip
module load python/3.11.5
python -V

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip

module load StdEnv/2023
module load rust/1.70.0
pip install --no-index transformers==4.36.2
pip install -r /home/cindyli/llama2/requirements-llama2.txt

python -V
pip list

echo "Llama2 original use from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python /home/cindyli/llama2/original_use/original_use_7b_hf.py
