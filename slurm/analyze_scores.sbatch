#!/bin/bash
#SBATCH -p shared                       # Shared partition (no GPU needed)
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=100G                      # 100GB RAM (need to load 84GB of files)
#SBATCH -t 0-01:00                      # 1 hour max
#SBATCH -o logs/analyze_%j.out
#SBATCH -e logs/analyze_%j.err
#SBATCH --job-name=sparc3_analyze

# ============================================================================
# SparC³ Code Review: Attribution Score Analysis
#
# This job performs comprehensive analysis of attribution scores to validate
# correctness before proceeding with the full experiment.
# ============================================================================

echo "======================================================================"
echo "SparC³ Attribution Score Analysis - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "======================================================================"

# Load Python module
module load python/3.10.13-fasrc01

# Activate virtual environment
VENV_PATH="/n/holylfs06/LABS/krajan_lab/Lab/kzheng/.venvs/gpu-cu121"
source ${VENV_PATH}/bin/activate

# Set cache directories
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"

# Change to project directory
cd /n/home03/kzheng/2881r-toxic-proj

# Run analysis
echo ""
echo "Running comprehensive attribution score analysis..."
echo ""

python3 analyze_scores_efficient.py

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Analysis Complete"
else
    echo "❌ Analysis Failed (exit code: $EXIT_CODE)"
fi
echo "======================================================================"
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
