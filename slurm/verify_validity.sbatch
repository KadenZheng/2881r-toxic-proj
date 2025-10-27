#!/bin/bash
#SBATCH -p shared                       # Shared partition (no GPU needed)
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=120G                      # 120GB RAM (need to load multiple 28GB files)
#SBATCH -t 0-00:30                      # 30 minutes max
#SBATCH -o logs/verify_%j.out
#SBATCH -e logs/verify_%j.err
#SBATCH --job-name=sparc3_verify

# ============================================================================
# SparC³ Scientific Validity Verification
#
# Comprehensive verification to ensure results are natural and not artificial
# ============================================================================

echo "======================================================================"
echo "SparC³ Scientific Validity Verification - Job ${SLURM_JOB_ID}"
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

# Run comprehensive verification
echo ""
echo "Running comprehensive scientific validity verification..."
echo ""

python3 verify_scientific_validity.py

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Verification Complete"
else
    echo "❌ Verification Failed (exit code: $EXIT_CODE)"
fi
echo "======================================================================"
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
