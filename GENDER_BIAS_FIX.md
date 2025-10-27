# Gender Bias Experiment - Fix Implementation

**Date**: October 27, 2025
**Status**: ✅ Fix Implemented - Pipeline Running with Validation Checkpoints

---

## 🚨 **What Went Wrong (BOLD Failure)**

### **The Problem**

**BOLD Experiment Results** (CATASTROPHIC FAILURE):
- Perplexity: 5.47 → **761.57** (+13,833%) ❌
- Gender Bias: 0.0671 → **0.1513** (+125%) ❌
- Model completely broken

### **Root Cause**

**Attribution Relevance Analysis**:
- R_general (C4): 5.76e+04
- R_gender (BOLD): 5.69e+04
- **Differential**: 0.07e+04 (only 1.2%!)

**Why**:
BOLD prompts were **biographical** ("Jacob Zachar is an American actor whose..."), NOT **stereotype-eliciting**.

**What happened**:
1. Biographical prompts activated general factual/knowledge circuits
2. Nearly identical circuits as C4 general text
3. Minimal differential signal (1.2%)
4. "Most negative R_diff" neurons were actually general-capability neurons
5. Pruning them destroyed the model

### **Contrast with Successful Toxicity**

**Toxicity Experiment**:
- R_general: 5.76e+04
- R_toxic: 4.32e+04
- **Differential**: 1.44e+04 (25%!) ✅

**Why it worked**:
- Toxic prompts activated **distinct toxic-specific circuits**
- Much lower total relevance than general
- Strong differential signal
- Pruning removed toxic circuits, preserved general

---

## ✅ **The Fix: MGS Stereotype-Elicitation-Prompt-Library**

### **New Dataset**

**Source**: wu981526092/Stereotype-Elicitation-Prompt-Library (HuggingFace)
- **200 gender stereotype prompts** (not biographical!)
- Specifically designed to elicit stereotypical completions
- Research-validated for bias detection

### **Prompt Comparison**

| Type | Example | Activates | Signal |
|------|---------|-----------|--------|
| **BOLD (Failed)** | "Jacob Zachar is an American actor whose..." | General biographical circuits | 1.2% diff ❌ |
| **MGS (Fix)** | "The nurse was very", "The engineer solved the problem. He" | Stereotype association circuits | Expected 20-35% diff ✅ |

### **Why MGS Should Work**

**MGS prompts are**:
- Incomplete sentences requiring stereotype-based completion
- Short (10-50 tokens, like toxic ~30 tokens)
- Trigger occupation-gender associations (nurse→female, engineer→male)
- Activate **different, narrower circuits** than general text

**Expected**:
- R_stereotype: 3.5-4.5e+04 (20-35% lower than general)
- Strong differential signal like toxicity
- Clear identification of stereotype-specific neurons

---

## 🛡️ **New Safety Feature: Validation Checkpoint**

### **CRITICAL Addition**

**New script**: `scripts/validate_differential.py`

**Purpose**: PREVENT repeating BOLD disaster by checking differential BEFORE pruning

**Validation Logic**:
```python
R_general = 5.76e+04
R_stereotype = ?
diff_pct = (R_general - R_stereotype) / R_general * 100

if diff_pct >= 15%:
    ✅ SAFE TO PROCEED - strong signal
else:
    ❌ STOP - too weak, will damage model
```

### **Thresholds**

| Differential | Interpretation | Action |
|--------------|----------------|--------|
| **≥ 20%** | Strong signal | ✅ Excellent - proceed confidently |
| **15-20%** | Moderate signal | ✅ Acceptable - proceed with monitoring |
| **10-15%** | Weak signal | ⚠️ Risky - consider better prompts |
| **< 10%** | Too weak | ❌ STOP - will fail like BOLD |

**Historical Evidence**:
- Toxicity (successful): 25% differential
- BOLD (failed): 1.2% differential
- **Threshold**: 15% minimum for safety

---

## 🔄 **New Pipeline with Validation**

### **Job 1: Prepare Stereotype Prompts** (RUNNING NOW)

**Job ID**: 43584567
**Status**: Running (using MGS, not BOLD)

**What it does**:
1. Load 200 MGS stereotype prompts
2. Generate completions with LLaMA-3-8B
3. Score with ModernBERT
4. Select top 93 highest-bias prompts (bias >0.7 target)
5. Save: `data/gender_bias_prompts.pkl`

**Expected**: ~15 minutes

### **Job 2: Compute Attribution**

```bash
sbatch slurm/compute_gender_attributions.sbatch
```

**What it does**:
- Compute LRP on selected stereotype prompts
- Save: `lrp_stereotype.pt`

**Expected**: ~20 minutes

### **Job 3: VALIDATION CHECKPOINT** ✅

```bash
sbatch slurm/validate_stereotype_differential.sbatch
```

**CRITICAL CHECK**:
1. Load R_general and R_stereotype
2. Calculate differential percentage
3. **GO/NO-GO decision**:
   - If ≥ 15%: ✅ Approve experiment
   - If < 15%: ❌ Stop, try WinoBias backup

**Expected**: ~5 minutes

### **Job 4: Run Experiment** (Only if validated!)

```bash
# ONLY run if validation passed!
sbatch slurm/run_gender_experiment.sbatch
```

**Expected**: ~1 hour

---

## 📊 **Expected Results (If Validation Passes)**

### **Differential Check**:
- R_general: 5.76e+04
- R_stereotype: 3.5-4.5e+04 (target)
- **Differential**: 1.3-2.3e+04 (20-35%)

### **After Pruning 100 Neurons**:
- Perplexity: 5.47 → 5.5-5.6 (< 2% increase) ✅
- Gender bias: 0.15 → 0.10-0.12 (20-35% reduction) ✅
- Model functional ✅

---

## 🔧 **Code Changes Summary**

**Modified**:
1. `src/data_prep.py`:
   - Added `load_mgs_stereotype_prompts()` function
   - Changed `prepare_gender_bias_prompts()` to use MGS instead of BOLD

2. `scripts/validate_differential.py`: NEW - validation checkpoint
3. `slurm/validate_stereotype_differential.sbatch`: NEW - validation job
4. `slurm/compute_gender_attributions.sbatch`: Output renamed to lrp_stereotype.pt
5. `slurm/run_gender_experiment.sbatch`: Input renamed to lrp_stereotype.pt

**Total changes**: ~100 lines

---

## 🎯 **Success Criteria**

**Validation Phase**:
- ✅ Differential ≥ 15% (preferably 20-35%)
- ✅ R_stereotype significantly < R_general
- ✅ Bias scores of selected prompts >0.7

**Experiment Phase** (only if validated):
- ✅ Perplexity < 5.6 (< 2% increase)
- ✅ Gender bias reduction 15-30%
- ✅ Model remains functional

---

## 🚀 **Current Status**

✅ Code modified to use MGS
✅ Validation checkpoint implemented
⏳ Job 1 running (MGS two-pass filtering)
⏳ Job 2 pending (attribution)
⏳ Job 3 pending (CRITICAL validation)
⏳ Job 4 pending (experiment - only if validated!)

---

## 📚 **Backup Plan**

**If validation fails** (differential < 15%):

**Option B**: WinoBias Conversion
- Convert WinoBias coreference prompts to generation format
- Truncate before pronouns to trigger stereotypes
- Higher quality, academic-validated dataset

**Option C**: Custom GPT-4 Generation
- Generate 500 stereotype-eliciting prompts
- Manual quality review
- Most control, highest quality

**Decision criteria**: If MGS gets 10-15% differential (borderline), try WinoBias. If MGS gets <10%, definitely need WinoBias or custom.

---

**Key Learning**: Biographical prompts ≠ Stereotype prompts. Must use prompts designed to trigger association circuits, not factual circuits.

**Implementation Status**: ✅ Fix complete, validation safeguards in place, pipeline running
