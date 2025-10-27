#!/bin/bash
#SBATCH -p shared                       # Shared partition (no GPU needed)
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=120G                      # 120GB RAM (need to load ~140GB attribution scores)
#SBATCH -t 0-01:00                      # 1 hour max
#SBATCH -o logs/overlap_%j.out
#SBATCH -e logs/overlap_%j.err
#SBATCH --job-name=overlap_analysis

# ============================================================================
# Circuit Overlap Analysis: Toxicity vs Gender Bias
#
# Analyzes whether toxic and gender-biased behaviors share common neural
# circuits or are processed independently.
#
# Phases:
# 1. Simple overlap (exact neuron matches)
# 2. Differential correlation (all 458k neurons)
# 4. Layer distribution (per-layer patterns)
#
# Expected runtime: ~30 minutes
# ============================================================================

echo "======================================================================"
echo "Circuit Overlap Analysis - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Memory: 120GB"
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

# Run overlap analysis
echo ""
echo "Running circuit overlap analysis..."
echo ""
echo "This will:"
echo "  1. Compare pruned neuron sets (toxicity vs gender)"
echo "  2. Compute correlation across all 458,752 neurons"
echo "  3. Analyze layer distribution patterns"
echo ""

python3 scripts/analyze_circuit_overlap.py

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Circuit Overlap Analysis Complete"
    echo ""
    echo "Results saved to: circuit_overlap_analysis.json"
    echo ""
    echo "Key outputs:"
    echo "  - Neuron overlap count and statistical significance"
    echo "  - Correlation: R_diff_toxic vs R_diff_gender"
    echo "  - Quadrant analysis: multi-bias vs specific neurons"
    echo "  - Layer distribution and hotspots"
else
    echo "❌ Analysis Failed (exit code: $EXIT_CODE)"
fi
echo "======================================================================"
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
