#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:2                    # Request 2× A100 40GB GPUs
#SBATCH -c 8                            # 8 CPU cores
#SBATCH --mem=128G                      # 128GB RAM
#SBATCH -t 0-01:00                      # 1 hour max
#SBATCH -o logs/attr_gender_%j.out
#SBATCH -e logs/attr_gender_%j.err
#SBATCH --job-name=attr_gender

# ============================================================================
# Gender Bias Attribution Computation
#
# Computes LRP attribution scores on gender-biased prompts.
# Reuses existing general attribution scores from C4.
#
# Expected runtime: ~20-30 minutes for ~90 prompts
# ============================================================================

echo "======================================================================"
echo "Gender Bias Attribution Computation - Job ${SLURM_JOB_ID}"
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

# Set paths
PROJECT_DIR="/n/home03/kzheng/2881r-toxic-proj"
DATA_DIR="/n/holylfs06/LABS/krajan_lab/Lab/kzheng/data/sparc3"
OUTPUT_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================================
# Verify gender bias prompts exist
# ============================================================================
GENDER_PROMPTS="${PROJECT_DIR}/data/gender_bias_prompts.pkl"

if [ ! -f "$GENDER_PROMPTS" ]; then
    echo "ERROR: Gender bias prompts not found at $GENDER_PROMPTS"
    echo "Please run prepare_gender_prompts.sbatch first"
    exit 1
fi

echo ""
echo "✓ Found gender bias prompts: $GENDER_PROMPTS"

# ============================================================================
# Compute gender bias attributions
# ============================================================================
echo ""
echo "======================================================================"
echo "Computing Gender Bias Attributions"
echo "======================================================================"

python scripts/compute_attributions.py \
    --method lrp \
    --samples ${GENDER_PROMPTS} \
    --output ${OUTPUT_DIR}/lrp_stereotype.pt \
    --model meta-llama/Meta-Llama-3-8B \
    --device auto \
    --dtype bfloat16 \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Gender attribution computation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ Gender Attribution Computation Complete"
echo "======================================================================"
echo "End time: $(date)"
echo ""
echo "Output file: ${OUTPUT_DIR}/lrp_gender.pt"
echo "File size:"
ls -lh ${OUTPUT_DIR}/lrp_gender.pt
echo ""
echo "Next step: Run gender bias experiment"
echo "  sbatch slurm/run_gender_experiment.sbatch"
echo "======================================================================"
