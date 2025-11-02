#!/bin/bash
#SBATCH -p gpu                          # GPU partition
#SBATCH --gres=gpu:2                    # Request 2× A100 40GB GPUs
#SBATCH -c 8                            # 8 CPU cores
#SBATCH --mem=256G                      # 256GB RAM (larger for loading multiple score files)
#SBATCH -t 0-04:00                      # 4 hours max
#SBATCH -o logs/exp_race_%j.out
#SBATCH -e logs/exp_race_%j.err
#SBATCH --job-name=exp_race

# ============================================================================
# SparC³ Racial Bias Removal Experiment
#
# This script runs the complete pipeline:
# 1. Load attribution scores
# 2. Compute differential and identify neurons
# 3. Prune neurons
# 4. Evaluate before and after
#
# Expected runtime: ~2-3 hours
# ============================================================================

echo "======================================================================"
echo "SparC³ Racial Bias Removal Experiment - Job ${SLURM_JOB_ID}"
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

# Set HuggingFace cache directories
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"

# Set PyTorch Hub cache (for classifiers)
export TORCH_HOME="${HOME}/.cache/torch"
mkdir -p ${TORCH_HOME}

# Set paths
PROJECT_DIR="/n/home03/kzheng/2881r-toxic-proj"
SCORES_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores"
RESULTS_DIR="/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results"

# Create output directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${PROJECT_DIR}/logs

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================================
# Verify required files exist
# ============================================================================
echo ""
echo "======================================================================"
echo "Verifying Required Files"
echo "======================================================================"

# Check attribution scores
REQUIRED_FILES=(
    "${SCORES_DIR}/lrp_general_seed0.pt"
    "${SCORES_DIR}/lrp_general_seed1.pt"
    "${SCORES_DIR}/lrp_general_seed2.pt"
    "${SCORES_DIR}/lrp_race.pt"
    "${PROJECT_DIR}/data/race_bias_prompts.pkl"
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
    echo "Please run compute_race_attributions.sbatch first."
    exit 1
fi

echo ""
echo "✓ All required files found"

# ============================================================================
# Run Full Experiment
# ============================================================================
echo ""
echo "======================================================================"
echo "Running Full Experiment"
echo "======================================================================"

# Set experiment parameters
NUM_NEURONS=100
LAYER_PATTERN="up_proj"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_OUTPUT_DIR="${RESULTS_DIR}/race_experiment_${TIMESTAMP}"
mkdir -p ${EXP_OUTPUT_DIR}

echo "Parameters:"
echo "  Neurons to prune: $NUM_NEURONS"
echo "  Layer pattern: $LAYER_PATTERN"
echo "  Output directory: $EXP_OUTPUT_DIR"
echo ""

# Run experiment
python scripts/run_race_experiment.py \
    --general_scores ${SCORES_DIR}/lrp_general_seed0.pt \
                     ${SCORES_DIR}/lrp_general_seed1.pt \
                     ${SCORES_DIR}/lrp_general_seed2.pt \
    --race_scores ${SCORES_DIR}/lrp_race.pt \
    --race_prompts ${PROJECT_DIR}/data/race_bias_prompts.pkl \
    --num_neurons ${NUM_NEURONS} \
    --layer_pattern ${LAYER_PATTERN} \
    --model meta-llama/Meta-Llama-3-8B \
    --device auto \
    --dtype bfloat16 \
    --output_dir ${EXP_OUTPUT_DIR} \
    --save_model

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment failed"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "✅ Experiment Complete"
echo "======================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $EXP_OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh ${EXP_OUTPUT_DIR}
echo ""
echo "To view results:"
echo "  cat ${EXP_OUTPUT_DIR}/experiment_results_race_*.json"
echo ""
echo "Next step: Three-way overlap analysis"
echo "  python scripts/analyze_threeway_overlap.py"
echo "======================================================================"

