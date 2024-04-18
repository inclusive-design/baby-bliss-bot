#!/bin/bash
#SBATCH --job-name=RAG-Mistral-7B-Instruct-v0.2
#SBATCH --time 2-00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --account=def-whkchun
#SBATCH --output=%x.o%j
 
module load StdEnv/2023
module load python/3.11.5

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip

module load StdEnv/2023 rust/1.70.0 arrow/15.0.1 gcc/12.3 cudacore/.12.2.2

pip install --no-index torch transformers tensorflow sentence_transformers accelerate==0.25.0 peft==0.5.0 bitsandbytes==0.42.0 datasets==2.17.0 trl
pip install langchain langchain-community unstructured chromadb

echo "=== Use RAG with Mistral-7b-instruct-1.0 from job $SLURM_JOB_ID on nodes $SLURM_JOB_NODELIST."
python ~/mistral/RAG/rag.py
