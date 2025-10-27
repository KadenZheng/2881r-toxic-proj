#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:2                    # Request 2× A100 40GB GPUs
#SBATCH -c 8                            # 8 CPU cores
#SBATCH --mem=128G                      # 128GB RAM
#SBATCH -t 0-06:00                      # 6 hours max
#SBATCH -o logs/attr_%j.out             # stdout to logs/
#SBATCH -e logs/attr_%j.err             # stderr to logs/
#SBATCH --job-name=sparc3_attr          # Job name

# ============================================================================
# SparC³ Attribution Computation
#
# This script computes LRP attribution scores for general and toxic samples.
# Expected runtime: ~3.5 hours for full dataset (128×3 + 93 samples)
# ============================================================================

echo "======================================================================"
echo "SparC³ Attribution Computation - Job ${SLURM_JOB_ID}"
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
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source ${VENV_PATH}/bin/activate

# Export HF token
HF_TOKEN_FILE="/n/holylfs06/LABS/krajan_lab/Lab/kzheng/.hf_token"
if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: HF token file not found at $HF_TOKEN_FILE"
    exit 1
fi
export HF_TOKEN=$(cat $HF_TOKEN_FILE)

# Set HuggingFace cache directories (fix /cache permission error)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"

# Set paths
PROJECT_DIR="/n/home03/kzheng/2881r-toxic-proj"
DATA_DIR="/n/holylfs06/LABS/krajan_lab/Lab/kzheng/data/sparc3"
OUTPUT_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================================
# 1. Prepare data (if not already done)
# ============================================================================
echo ""
echo "======================================================================"
echo "STEP 1: Preparing Data"
echo "======================================================================"

if [ ! -f "${DATA_DIR}/c4_general_seed0.pkl" ]; then
    echo "Preparing C4 and toxic datasets..."
    python scripts/prepare_data.py \
        --c4_samples 128 \
        --seq_len 2048 \
        --toxic_count 93 \
        --seeds 0 1 2

    # Move data files to proper location
    mkdir -p ${DATA_DIR}
    mv data/*.pkl ${DATA_DIR}/ 2>/dev/null || true

    echo "✓ Data preparation complete"
else
    echo "✓ Data files already exist, skipping preparation"
fi

# ============================================================================
# 2. Compute general attributions (3 seeds)
# ============================================================================
echo ""
echo "======================================================================"
echo "STEP 2: Computing General Attributions (3 seeds)"
echo "======================================================================"

for SEED in 0 1 2; do
    echo ""
    echo "--- Seed $SEED ---"

    python scripts/compute_attributions.py \
        --method lrp \
        --samples ${DATA_DIR}/c4_general_seed${SEED}.pkl \
        --output ${OUTPUT_DIR}/lrp_general_seed${SEED}.pt \
        --model meta-llama/Meta-Llama-3-8B \
        --device auto \
        --dtype bfloat16 \
        --verbose

    if [ $? -ne 0 ]; then
        echo "ERROR: Attribution computation failed for seed $SEED"
        exit 1
    fi
done

echo "✓ General attributions complete for all seeds"

# ============================================================================
# 3. Compute toxic attributions
# ============================================================================
echo ""
echo "======================================================================"
echo "STEP 3: Computing Toxic Attributions"
echo "======================================================================"

python scripts/compute_attributions.py \
    --method lrp \
    --samples ${DATA_DIR}/realtoxicityprompts_toxic.pkl \
    --output ${OUTPUT_DIR}/lrp_toxic.pt \
    --model meta-llama/Meta-Llama-3-8B \
    --device auto \
    --dtype bfloat16 \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Toxic attribution computation failed"
    exit 1
fi

echo "✓ Toxic attributions complete"

# ============================================================================
# 4. Verify outputs
# ============================================================================
echo ""
echo "======================================================================"
echo "STEP 4: Verifying Outputs"
echo "======================================================================"

echo "Generated files:"
ls -lh ${OUTPUT_DIR}/*.pt

# Check file sizes
for FILE in ${OUTPUT_DIR}/*.pt; do
    SIZE=$(du -h $FILE | cut -f1)
    echo "  $(basename $FILE): $SIZE"
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "✅ Attribution Computation Complete"
echo "======================================================================"
echo "End time: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "======================================================================"
