#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:2                    # Request 2× A100 40GB GPUs
#SBATCH -c 8                            # 8 CPU cores
#SBATCH --mem=128G                      # 128GB RAM
#SBATCH -t 0-02:00                      # 2 hours max
#SBATCH -o logs/exp_gender_%j.out
#SBATCH -e logs/exp_gender_%j.err
#SBATCH --job-name=exp_gender

# ============================================================================
# SparC³ Gender Bias Removal Experiment
#
# Complete pipeline:
# 1. Load attribution scores (general + gender)
# 2. Compute differential and identify neurons
# 3. Prune neurons
# 4. Evaluate before and after (perplexity + gender bias)
#
# Expected runtime: ~1 hour
# ============================================================================

echo "======================================================================"
echo "SparC³ Gender Bias Removal Experiment - Job ${SLURM_JOB_ID}"
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
SCORES_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores"
RESULTS_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results"

# Create output directories
mkdir -p ${RESULTS_DIR}
mkdir -p logs

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================================
# Verify required files exist
# ============================================================================
echo ""
echo "======================================================================"
echo "Verifying Required Files"
echo "======================================================================"

REQUIRED_FILES=(
    "${SCORES_DIR}/lrp_general_seed0.pt"
    "${SCORES_DIR}/lrp_general_seed1.pt"
    "${SCORES_DIR}/lrp_general_seed2.pt"
    "${SCORES_DIR}/lrp_stereotype.pt"
    "${PROJECT_DIR}/data/gender_bias_prompts.pkl"
)

ALL_EXIST=true
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "❌ Missing: $FILE"
        ALL_EXIST=false
    else
        SIZE=$(du -h $FILE | cut -f1)
        echo "✓ Found: $(basename $FILE) ($SIZE)"
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "ERROR: Some required files are missing."
    echo "Run preparation steps first:"
    echo "  1. sbatch slurm/prepare_gender_prompts.sbatch"
    echo "  2. sbatch slurm/compute_gender_attributions.sbatch"
    exit 1
fi

echo ""
echo "✓ All required files found"

# ============================================================================
# Run Gender Bias Experiment
# ============================================================================
echo ""
echo "======================================================================"
echo "Running Gender Bias Experiment"
echo "======================================================================"

# Set experiment parameters
NUM_NEURONS=100
LAYER_PATTERN="up_proj"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_OUTPUT_DIR="${RESULTS_DIR}/gender_experiment_${TIMESTAMP}"
mkdir -p ${EXP_OUTPUT_DIR}

echo "Parameters:"
echo "  Neurons to prune: $NUM_NEURONS"
echo "  Layer pattern: $LAYER_PATTERN"
echo "  Output directory: $EXP_OUTPUT_DIR"
echo ""

# Run experiment
python scripts/run_gender_experiment.py \
    --general_scores ${SCORES_DIR}/lrp_general_seed0.pt \
                     ${SCORES_DIR}/lrp_general_seed1.pt \
                     ${SCORES_DIR}/lrp_general_seed2.pt \
    --gender_scores ${SCORES_DIR}/lrp_stereotype.pt \
    --gender_prompts ${PROJECT_DIR}/data/gender_bias_prompts.pkl \
    --num_neurons ${NUM_NEURONS} \
    --layer_pattern ${LAYER_PATTERN} \
    --model meta-llama/Meta-Llama-3-8B \
    --device auto \
    --dtype bfloat16 \
    --output_dir ${EXP_OUTPUT_DIR} \
    --save_model

if [ $? -ne 0 ]; then
    echo "ERROR: Gender bias experiment failed"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "✅ Gender Bias Experiment Complete"
echo "======================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $EXP_OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh ${EXP_OUTPUT_DIR}
echo ""
echo "To view results:"
echo "  cat ${EXP_OUTPUT_DIR}/experiment_results_gender_*.json"
echo "======================================================================"
