#!/bin/bash
#SBATCH --job-name=stylegan3
#SBATCH --time 24-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --account={replace_with_your_account}
#SBATCH --output=%x.o%j
 
pip install --no-index --upgrade pip
module load python/3.8.2
python -V

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

module load scipy-stack/2021a
module load cuda/11.1.1

# Check CUDA Toolkit version
nvcc -V

# Check gcc version
which gcc

pip install -r /home/cindyli/stylegan3/stylegan3/requirements.txt

# Check the install packages
pip list

export CUDA_LAUNCH_BLOCKING=1

echo "Hello from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
# nvidia-smi
python /home/cindyli/stylegan3/stylegan3/train.py --outdir=/home/cindyli/stylegan3/training-runs --cfg=stylegan3-r --data=/home/cindyli/stylegan3/datasets/bliss-256x256.zip --gpus=1 --batch=32 --gamma=2 --batch-gpu=8 --snap=10
