#!/bin/bash
#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem=80G
#SBATCH -t 0-00:15
#SBATCH -o logs/validate_diff_%j.out
#SBATCH -e logs/validate_diff_%j.err
#SBATCH --job-name=validate_diff

# ============================================================================
# CRITICAL VALIDATION: Check Differential Signal Strength
#
# This script validates that stereotype prompts create strong enough
# differential signal (≥15%) before proceeding to pruning.
#
# Prevents catastrophic failure like BOLD (1.2% differential).
# ============================================================================

echo "======================================================================"
echo "Differential Signal Validation - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Start time: $(date)"
echo "======================================================================"

# Load Python
module load python/3.10.13-fasrc01

# Activate venv
source /n/holylfs06/LABS/krajan_lab/Lab/kzheng/.venvs/gpu-cu121/bin/activate

# Set cache
export HF_HOME="${HOME}/.cache/huggingface"

# Change to project
cd /n/home03/kzheng/2881r-toxic-proj

# Run validation
echo ""
echo "Validating differential signal..."
echo ""

python scripts/validate_differential.py \
    --general_scores /n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt \
                     /n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed1.pt \
                     /n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed2.pt \
    --behavior_scores /n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_stereotype.pt \
    --min_differential 15.0 \
    --behavior_name stereotype

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ VALIDATION PASSED - Safe to proceed with pruning"
    echo ""
    echo "Next step:"
    echo "  sbatch slurm/run_gender_experiment.sbatch"
else
    echo "❌ VALIDATION FAILED - DO NOT proceed with pruning"
    echo ""
    echo "Differential signal too weak. Try:"
    echo "  1. Different prompts (WinoBias conversion)"
    echo "  2. Custom GPT-4 generated prompts"
    echo "  3. More selective filtering (higher bias threshold)"
fi
echo "======================================================================"
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
