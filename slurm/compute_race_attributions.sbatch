#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:2                    # Request 2× A100 40GB GPUs
#SBATCH -c 8                            # 8 CPU cores
#SBATCH --mem=128G                      # 128GB RAM
#SBATCH -t 0-02:00                      # 2 hours max
#SBATCH -o logs/attr_race_%j.out
#SBATCH -e logs/attr_race_%j.err
#SBATCH --job-name=attr_race

# ============================================================================
# Racial Attribution Computation
#
# Computes LRP scores for racial bias prompts (93 prompts, 1 seed)
# Reuses existing R_general scores for efficiency
#
# Expected runtime: ~54 minutes
# ============================================================================

echo "======================================================================"
echo "Racial Attribution Computation - Job ${SLURM_JOB_ID}"
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
if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: HF token file not found at $HF_TOKEN_FILE"
    exit 1
fi
export HF_TOKEN=$(cat $HF_TOKEN_FILE)

# Set HuggingFace cache directories
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"

# Set paths
PROJECT_DIR="/n/home03/kzheng/2881r-toxic-proj"
DATA_FILE="${PROJECT_DIR}/data/race_bias_prompts.pkl"
OUTPUT_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_DIR}/logs

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================================
# Verify data file exists
# ============================================================================
echo ""
echo "======================================================================"
echo "Verifying Data File"
echo "======================================================================"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Race prompts file not found: $DATA_FILE"
    echo "Please run: python scripts/prepare_race_prompts.py"
    exit 1
fi

SIZE=$(du -h $DATA_FILE | cut -f1)
echo "✓ Found: $(basename $DATA_FILE) ($SIZE)"

# ============================================================================
# Compute racial attribution
# ============================================================================
echo ""
echo "======================================================================"
echo "Computing Racial Attribution (LRP)"
echo "======================================================================"
echo "Samples: 93 race bias prompts"
echo "Output: ${OUTPUT_DIR}/lrp_race.pt"
echo ""

python scripts/compute_attributions.py \
    --method lrp \
    --samples ${DATA_FILE} \
    --output ${OUTPUT_DIR}/lrp_race.pt \
    --model meta-llama/Meta-Llama-3-8B \
    --device auto \
    --dtype bfloat16 \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Attribution computation failed"
    exit 1
fi

# ============================================================================
# Verify output
# ============================================================================
echo ""
echo "======================================================================"
echo "Verifying Output"
echo "======================================================================"

if [ -f "${OUTPUT_DIR}/lrp_race.pt" ]; then
    SIZE=$(du -h ${OUTPUT_DIR}/lrp_race.pt | cut -f1)
    echo "✓ Created: lrp_race.pt ($SIZE)"
else
    echo "ERROR: Output file not created"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "✅ Racial Attribution Complete"
echo "======================================================================"
echo "End time: $(date)"
echo "Output: ${OUTPUT_DIR}/lrp_race.pt"
echo ""
echo "Next step: Run full experiment"
echo "  sbatch slurm/run_race_experiment.sbatch"
echo "======================================================================"

