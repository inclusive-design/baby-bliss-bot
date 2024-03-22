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

module load StdEnv/2023 rust/1.70.0 arrow/14.0.1 gcc/12.3
pip install --no-index transformers==4.36.2 accelerate==0.25.0 peft==0.5.0 bitsandbytes==0.42.0 tensorboard
pip install datasets==2.17.0 trl
pip install -r /home/cindyli/llama2/requirements-llama2.txt

echo "=== Fine-tuning Llama2 from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python /home/cindyli/llama2/finetune/finetune_7b_hf.py
