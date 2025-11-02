#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:2                    # Request 2× A100 40GB GPUs
#SBATCH -c 8                            # 8 CPU cores
#SBATCH --mem=256G                      # 256GB RAM (increased for loading multiple 28GB score files)
#SBATCH -t 0-01:30                      # 1.5 hours max
#SBATCH -o logs/validate_racial_%j.out
#SBATCH -e logs/validate_racial_%j.err
#SBATCH --job-name=validate_racial

# ============================================================================
# CRITICAL VALIDATION: Racial Differential Signal Test
#
# This is a GATE - determines if StereoSet is suitable for race experiment.
# Must achieve ≥15% differential to proceed.
#
# Expected runtime: ~20-30 minutes
# ============================================================================

echo "======================================================================"
echo "Racial Differential Validation - Job ${SLURM_JOB_ID}"
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
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat $HF_TOKEN_FILE)
fi

# Set HuggingFace cache directories
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"

# Set paths
PROJECT_DIR="/n/home03/kzheng/2881r-toxic-proj"
cd ${PROJECT_DIR}

# Create logs directory
mkdir -p logs

echo ""
echo "======================================================================"
echo "Running Differential Validation"
echo "======================================================================"

# Run validation
python scripts/validate_racial_differential.py \
    --samples data/race_bias_prompts.pkl \
    --n_validation 50 \
    --model meta-llama/Meta-Llama-3-8B \
    --min_differential 15.0

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ VALIDATION PASSED - Differential signal is strong enough"
    echo "   Safe to proceed with full racial bias experiment"
    echo ""
    echo "Next step:"
    echo "  sbatch slurm/compute_race_attributions.sbatch"
else
    echo ""
    echo "❌ VALIDATION FAILED - Differential signal too weak"
    echo "   Do NOT proceed with current prompts"
    echo ""
    echo "Recommended:"
    echo "  Switch to BOLD two-pass filtering"
    echo "  Run: python scripts/prepare_race_prompts_twopass.py"
fi

echo "======================================================================"

exit $EXIT_CODE

