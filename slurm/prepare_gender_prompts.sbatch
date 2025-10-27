#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:1                    # Request 1 GPU (enough for generation)
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=64G                       # 64GB RAM
#SBATCH -t 0-01:00                      # 1 hour max
#SBATCH -o logs/prep_gender_%j.out
#SBATCH -e logs/prep_gender_%j.err
#SBATCH --job-name=prep_gender

# ============================================================================
# Gender Bias Prompt Preparation
#
# Two-pass filtering to select prompts that elicit gender-biased completions:
# 1. Load ~500 BOLD gender candidate prompts
# 2. Generate completions with LLaMA-3-8B
# 3. Score completions for gender bias
# 4. Select top ~90 most bias-inducing prompts
#
# Expected runtime: ~30 minutes
# ============================================================================

echo "======================================================================"
echo "Gender Bias Prompt Preparation - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "======================================================================"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01

# Activate virtual environment
VENV_PATH="/n/holylfs06/LABS/krajan_lab/Lab/kzheng/.venvs/gpu-cu121"
source ${VENV_PATH}/bin/activate

# Export HF token
HF_TOKEN_FILE="/n/holylfs06/LABS/krajan_lab/Lab/kzheng/.hf_token"
export HF_TOKEN=$(cat $HF_TOKEN_FILE)

# Set cache directories
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"
export TORCH_HOME="${HOME}/.cache/torch"

# Change to project directory
cd /n/home03/kzheng/2881r-toxic-proj

# Create data directory
mkdir -p data

# Run two-pass filtering
echo ""
echo "Running two-pass filtering..."
echo ""

python scripts/prepare_gender_prompts.py \
    --target_count 93 \
    --max_candidates 500 \
    --model meta-llama/Meta-Llama-3-8B \
    --output data/gender_bias_prompts.pkl

if [ $? -ne 0 ]; then
    echo "ERROR: Gender prompt preparation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ Preparation Complete"
echo "======================================================================"
echo "End time: $(date)"
echo ""
echo "Output file: data/gender_bias_prompts.pkl"
echo ""
echo "Next step: Compute attributions"
echo "  sbatch slurm/compute_gender_attributions.sbatch"
echo "======================================================================"
