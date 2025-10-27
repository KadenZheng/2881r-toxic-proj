#!/bin/bash
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=100G
#SBATCH -t 0-00:30
#SBATCH -o logs/debug_gender_%j.out
#SBATCH -e logs/debug_gender_%j.err
#SBATCH --job-name=debug_gender

module load python/3.10.13-fasrc01
source /n/holylfs06/LABS/krajan_lab/Lab/kzheng/.venvs/gpu-cu121/bin/activate
export HF_HOME="${HOME}/.cache/huggingface"

cd /n/home03/kzheng/2881r-toxic-proj
python3 debug_gender_failure.py
