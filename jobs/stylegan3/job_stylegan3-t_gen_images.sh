#!/bin/bash
#SBATCH --job-name=stylegan3-t-gen-images
#SBATCH --time 10-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
 
pip install --no-index --upgrade pip
module load python/3.8.2
python -V

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

module load scipy-stack/2021a
module load cuda/11.1.1

echo "Start to check CUDA Toolkit version"
nvcc -V
echo "End of checking CUDA Toolkit version"

echo "Start to check gcc version"
which gcc
echo "End of checking gcc version"

pip install -r /home/cindyli/stylegan3/stylegan3/requirements.txt

pip list

export CUDA_LAUNCH_BLOCKING=1

echo "Hello from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python /home/cindyli/stylegan3/stylegan3/gen_images.py --outdir=/home/cindyli/stylegan3/out-stylegan3-t --trunc=1 --seeds=2 --network=/home/cindyli/stylegan3/training-runs/00016-stylegan3-t-bliss-256x256-gpus1-batch32-gamma2/network-snapshot-004160.pkl
