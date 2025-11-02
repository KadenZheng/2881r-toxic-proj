# SparC³ Reproduction: Multi-Bias Suppression in LLMs

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/your-repo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Reproducing and Extending Attribution-Guided Pruning for Multi-Bias Suppression in Large Language Models**

This repository contains code to reproduce the SparC³ framework \[[Hatefi et al., 2025](https://arxiv.org/abs/2506.13727)\] on LLaMA-3-8B and extend it to gender and racial biases.

## Overview

We faithfully reproduce the SparC³ toxicity suppression result (17.3% reduction, <1% perplexity impact) and extend the framework to gender (14.1% reduction) and racial (11.3% reduction) stereotypes. Our novel three-way circuit overlap analysis discovers 5 core universal bias neurons affecting all three behaviors (enrichment: 1,052,267×) and confirms that social biases (gender, race) cluster separately from toxicity.

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-repo/sparc3-multibias
cd sparc3-multibias

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (required for LLaMA-3)
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

### 2. Prepare Data

```bash
# Prepare C4 general samples + toxic prompts
python scripts/prepare_data.py --c4_samples 128 --toxic_count 93 --seeds 0 1 2

# Prepare gender prompts (two-pass filtering)
python scripts/prepare_gender_prompts.py --max_prompts 93

# Prepare race prompts (StereoSet)
python scripts/prepare_race_prompts.py --max_prompts 93
```

### 3. Run Experiments

**Toxicity Suppression** (reproduce paper):
```bash
# Compute attribution
python scripts/compute_attributions.py \
    --method lrp \
    --samples data/c4_general_seed0.pkl \
    --output scores/lrp_general_seed0.pt

# Run experiment
python scripts/run_full_experiment.py \
    --general_scores scores/lrp_general_seed*.pt \
    --toxic_scores scores/lrp_toxic.pt \
    --toxic_prompts data/realtoxicityprompts_toxic.pkl \
    --num_neurons 100 \
    --output_dir results/toxic_experiment/
```

**Gender & Race** (follow same pattern with respective scripts)

### 4. Analyze Results

```bash
# Three-way overlap analysis
python scripts/analyze_threeway_overlap.py

# Generate figures for paper
python scripts/generate_figures.py
```

## Repository Structure

```
.
├── README.md                   # This file
├── MAIN.md                     # Technical documentation
├── requirements.txt            # Python dependencies
├── src/                        # Core implementation
│   ├── attribution.py          # LRP & Wanda methods
│   ├── pruning.py              # Differential & neuron selection
│   ├── data_prep.py            # Dataset loading
│   └── evaluation.py           # Perplexity & bias scoring
├── scripts/                    # Experiment scripts
│   ├── prepare_data.py         # Data preparation
│   ├── compute_attributions.py # Attribution computation
│   ├── run_*_experiment.py     # Experiment pipelines (toxic/gender/race)
│   ├── analyze_threeway_overlap.py  # Circuit overlap analysis
│   └── generate_figures.py     # Publication figures
├── slurm/                      # GPU cluster job scripts
├── paper/                      # LaTeX paper
│   ├── reproduction_paper.tex
│   ├── references.bib
│   └── figures/                # Paper figures (fig1-3)
├── visualizations/             # Generated figures (fig1-6)
├── data/                       # Generated datasets (gitignored)
├── results/                    # Experiment outputs (gitignored)
└── archive/                    # Internal documentation (gitignored)
```

## Reproducing Paper Results

### Expected Outputs

| Experiment | Bias Reduction | PPL Change | Differential | Runtime (2×A100) |
|------------|----------------|------------|--------------|------------------|
| Toxicity   | -17.3%         | +0.80%     | 27.2%        | ~4 hours         |
| Gender     | -14.1%         | +0.95%     | 39.9%        | ~6 hours         |
| Race       | -11.3%         | +2.57%     | 37.6%        | ~4 hours         |

**Three-Way Analysis**:
- Triple overlap: 5 neurons (layers 15, 19, 28, 29, 30)
- Pairwise: Toxic-Gender 11%, Toxic-Race 6%, Gender-Race 15%
- Correlation: r(gender,race)=0.442 > r(toxic,race)=0.370

### Step-by-Step Reproduction

**Phase 1: Data Preparation** (~10 minutes)
```bash
python scripts/prepare_data.py --c4_samples 128 --toxic_count 93 --seeds 0 1 2
python scripts/prepare_gender_prompts.py
python scripts/prepare_race_prompts.py
```

**Phase 2: Attribution Computation** (~12 hours GPU time)
```bash
# General (3 seeds): ~3.5 hours each
for seed in 0 1 2; do
    python scripts/compute_attributions.py \
        --method lrp \
        --samples data/c4_general_seed${seed}.pkl \
        --output scores/lrp_general_seed${seed}.pt
done

# Toxic: ~54 minutes
python scripts/compute_attributions.py \
    --method lrp \
    --samples data/realtoxicityprompts_toxic.pkl \
    --output scores/lrp_toxic.pt

# Gender: ~54 minutes
python scripts/compute_attributions.py \
    --method lrp \
    --samples data/gender_bias_prompts.pkl \
    --output scores/lrp_gender.pt

# Race: ~54 minutes
python scripts/compute_attributions.py \
    --method lrp \
    --samples data/race_bias_prompts.pkl \
    --output scores/lrp_race.pt
```

**Phase 3: Experiments** (~6 hours GPU time)
```bash
# Toxicity
python scripts/run_full_experiment.py \
    --general_scores scores/lrp_general_seed{0,1,2}.pt \
    --toxic_scores scores/lrp_toxic.pt \
    --toxic_prompts data/realtoxicityprompts_toxic.pkl \
    --output_dir results/toxic/

# Gender  
python scripts/run_gender_experiment.py \
    --general_scores scores/lrp_general_seed{0,1,2}.pt \
    --gender_scores scores/lrp_gender.pt \
    --gender_prompts data/gender_bias_prompts.pkl \
    --output_dir results/gender/

# Race
python scripts/run_race_experiment.py \
    --general_scores scores/lrp_general_seed{0,1,2}.pt \
    --race_scores scores/lrp_race.pt \
    --race_prompts data/race_bias_prompts.pkl \
    --output_dir results/race/
```

**Phase 4: Analysis** (~30 minutes)
```bash
# Three-way overlap
python scripts/analyze_threeway_overlap.py

# Generate figures
python scripts/generate_figures.py
```

## Hardware Requirements

**Minimum**:
- 2× GPU with 40GB VRAM (e.g., A100)
- 256GB RAM
- 200GB storage

**Tested On**:
- Harvard FASRC cluster
- 2× NVIDIA A100 40GB
- ~22 GPU hours total

## Key Implementation Details

**Attribution Method**: Layer-wise Relevance Propagation ($\epsilon$-LRP) via [LXT library](https://github.com/rachtibat/LRP-eXplains-Transformers)

**Differential Attribution** (Equation 7 from paper):
```
R_diff = R_general - R_behavior
```
Neurons with most negative R_diff are behavior-specific.

**Architecture Adaptation**: LLaMA's gated MLP (up\_proj, gate\_proj, down\_proj) vs. OPT's standard MLP (fc1, fc2). When pruning neuron i, we zero all three components.

**Differential Validation**: Before full attribution (~50 GPU hours), test 50 prompts. Proceed only if differential ≥15%. This threshold prevents unsuitable dataset selection.

## Citation

If you use this code or reproduce our results, please cite:

```bibtex
@article{zheng2025sparc3reproduction,
  title={Reproducing and Extending Attribution-Guided Pruning: Multi-Bias Suppression in Large Language Models},
  author={Zheng, Kaden and Zen, Maxwell},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}

@article{hatefi2025sparc3,
  title={Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs},
  author={Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and others},
  journal={arXiv preprint arXiv:2506.13727},
  year={2025}
}
```

## License

MIT License - see LICENSE file

## Contact

- Kaden Zheng (kadenzheng@college.harvard.edu)
- Maxwell Zen (maxwellzen@college.harvard.edu)

## Acknowledgments

This work reproduces and extends the SparC³ framework by Hatefi et al. (2025). We thank the authors for their excellent paper. Computational resources provided by Harvard FAS Research Computing.

## Troubleshooting

**Out of Memory**: Increase GPU memory allocation or reduce batch size in attribution computation.

**Differential <15%**: Dataset may be unsuitable. Try two-pass filtering (see `scripts/prepare_gender_prompts.py` for example).

**Missing HF Token**: Set `HF_TOKEN` environment variable or create `.env` file.

For detailed technical documentation, see `MAIN.md`.

