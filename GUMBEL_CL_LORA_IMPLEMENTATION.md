# Gumbel CL-LoRA Implementation Summary

## Overview

This document describes the implementation of the Gumbel CL-LoRA formulation for continual learning on CIFAR-100. The implementation integrates sparse Gumbel-Sparsemax task selection with the existing SD-LoRA framework to enable **sublinear parameter growth** during continual learning.

## Mathematical Formulation Implemented

### 1. Decoupled Adapter Structure (Equation 1)

```
ΔW_t = Σ_{i=1}^{t} β_i * α_i * (A_i B_i / ||A_i B_i||_F)
```

**Implementation**: `backbone/lora.py:178-243` (_LoRA_qkv_timm_train.forward)

- **Direction normalization**: Divides by `||w_b|| * ||w_a|| + 1e-8` for numerical stability (lines 214-215)
- **Magnitude parameters** (`α_i`): Stored in `GumbelGate.alpha` (per-task)
- **Gate weights** (`β_i`): Computed via Gumbel-Sparsemax (line 225-230)
- **Pruned adapter handling**: Skipped entirely before normalization to prevent NaN (lines 186-187)

### 2. Gumbel-Sparsemax Sparse Gating (Equation 2)

```
β = sparsemax((l + G) / τ), where G ~ Gumbel(0,1)
```

**Implementation**: `utils/gumbel_utils.py:32-86` (sparsemax), `utils/gumbel_utils.py:89-129` (gumbel_sparsemax)

- **Sparsemax projection**: Yields exact zeros (unlike softmax)
- **Gumbel noise**: Added during training for exploration
- **Temperature τ**: Annealed from 5.0 → 0.5 during training
- **Numerical stability**: -inf replaced with -1e9, epsilon added to divisions

### 3. Sparsity Regularization (Equation 3)

```
L_total = L_task + λ_sparsity * Ω_sparsity(β)
Ω_sparsity = -Σ β_i log(β_i + ε)
```

**Implementation**: `models/sdlora.py:259-274` (_update_representation)

- **Sparsity loss collection**: Collected from all LoRA layers per forward pass
- **Total loss**: `loss = loss_clf + lambda_sparsity * sparsity_loss_total`
- **Default λ_sparsity**: 0.001
- **Note**: Sparsity loss is 0 when only 1 adapter exists (Task 1) - this is correct behavior

### 4. Conditional Growth Mechanism (Equation 4)

```
If β_t > δ: Keep adapter, N ← N+1
If β_t ≤ δ: Prune adapter, N unchanged
```

**Implementation**: `models/sdlora.py:322-383` (_conditional_growth_decision)

- **Threshold δ**: 1e-6 (trusts sparsemax's natural zeros)
- **Pruning mask**: Stored in `GumbelGate.pruning_mask` and persisted to disk
- **Beta logging**: Saved to `CF100/beta_values_task_*.pt` for analysis
- **Mask persistence**: Loaded at initialization to prevent "reappearing" adapters

## File Structure

### New Files

1. **`utils/gumbel_utils.py`** (247 lines)
   - `sample_gumbel()`: Gumbel(0,1) noise sampling
   - `sparsemax()`: Sparsemax projection with numerical stability safeguards
   - `gumbel_sparsemax()`: Combines Gumbel noise + Sparsemax
   - `sparsity_loss()`: Entropy-based sparsity regularization
   - `TemperatureScheduler`: Exponential temperature annealing

### Modified Files

2. **`backbone/lora.py`**
   - **Added**: `GumbelGate` class (lines 311-449)
     - Manages α (magnitude) and β (gate) parameters per task
     - Implements Gumbel-Sparsemax forward pass
     - Tracks pruning mask for conditional growth
     - Persists mask across tasks

   - **Modified**: `_LoRA_qkv_timm_train` (lines 133-247)
     - **CRITICAL FIX**: Skips pruned adapters before normalization (lines 186-187)
     - Collects normalized adapter outputs with epsilon for stability
     - Applies Gumbel-Sparsemax gating
     - Stores sparsity loss for backprop

   - **Modified**: `_LoRA_qkv_timm_eval` (lines 249-323)
     - Updated for GumbelGate (no Gumbel noise in eval)

   - **Modified**: `LoRA_ViT_timm.__init__` (lines 519-532)
     - Instantiates `GumbelGate` instead of `ParameterWrapper`
     - **Loads pruning mask** from previous tasks if exists
     - Passes `gumbel_gate` to all LoRA layers

3. **`models/sdlora.py`**
   - **Added**: Import `TemperatureScheduler` (line 13)

   - **Modified**: `_update_representation` (lines 217-320)
     - Initializes temperature scheduler
     - Anneals τ during training
     - Collects sparsity loss from LoRA layers
     - Adds sparsity regularization to total loss
     - Logs detailed loss breakdown (clf + sparse + tau)

   - **Modified**: `_conditional_growth_decision` (lines 322-383)
     - Simplified to trust sparsemax (threshold=1e-6)
     - Evaluates β values after training
     - Prunes adapters where sparsemax → 0
     - Saves pruning mask for persistence
     - Logs active/pruned adapter statistics

4. **`exps/sdlora_c100.json`**
   - **Added**: Gumbel CL-LoRA hyperparameters
     ```json
     "gumbel_tau_init": 5.0,
     "gumbel_tau_final": 0.5,
     "gumbel_anneal_rate": 0.999,
     "lambda_sparsity": 0.001,
     "growth_threshold": 1e-6
     ```

## How It Works

### Training Pipeline (Task t > 0)

1. **Initialization** (`models/sdlora.py:224-226`)
   - Create `TemperatureScheduler` with τ_init=5.0, τ_final=0.5
   - Initialize GumbelGate with learnable α and logit parameters
   - Load pruning mask from previous tasks if exists

2. **Forward Pass** (`backbone/lora.py:183-243`)
   - For each previous task i ∈ [0, t-1]:
     - **Check pruning mask first** (line 186)
     - If pruned (mask=0), skip entirely to avoid NaN
     - If active (mask=1):
       - Load saved LoRA weights A_i, B_i
       - Compute normalized output: `LoRA_i(x) / (||A_i|| * ||B_i|| + 1e-8)`
   - Apply Gumbel-Sparsemax gating:
     - Sample Gumbel noise (if training)
     - Compute β = sparsemax((logits + noise) / τ)
     - Weighted sum: `Σ β_i * α_i * LoRA_i(x)`
   - Add current task adapter: `α_t * LoRA_t(x)`

3. **Loss Computation** (`models/sdlora.py:253-274`)
   - Classification loss: Cross-entropy on current task
   - Sparsity loss: Collect from all LoRA layers (returns 0 if only 1 adapter)
   - Total loss: `L = L_clf + λ * L_sparsity`

4. **Temperature Annealing** (`models/sdlora.py:241`)
   - Each iteration: `τ ← τ_final + (τ_init - τ_final) * anneal_rate^step`
   - Encourages β to become more sparse over time

5. **Conditional Growth** (`models/sdlora.py:322-383`)
   - After training, evaluate β values for all tasks
   - Trust sparsemax: if β_i ≈ 0 (< 1e-6), adapter is not useful
   - Set mask=0 for pruned adapters
   - Save mask to `CF100/pruning_mask.pt` for next task

### Evaluation Mode

- No Gumbel noise (deterministic)
- Uses learned β values (no sampling)
- Pruned adapters automatically excluded (skipped in forward pass)

## Hyperparameter Tuning Guide

### Gumbel-Sparsemax Parameters

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `gumbel_tau_init` | 5.0 | Initial temperature | Higher = more exploration, softer selection |
| `gumbel_tau_final` | 0.5 | Final temperature | Lower = sharper selection, closer to hard |
| `gumbel_anneal_rate` | 0.999 | Decay rate per step | Higher = slower annealing |
| `lambda_sparsity` | 0.001 | Sparsity regularization weight | Higher = sparser β, fewer active adapters |
| `growth_threshold` | 1e-6 | Pruning threshold | Set very low to trust sparsemax's learned zeros |

### Expected Behavior

**High τ (early training)**:
- β is more uniform (e.g., [0.3, 0.3, 0.2, 0.2])
- Multiple adapters contribute
- Higher exploration

**Low τ (late training)**:
- β is sparse (e.g., [0.7, 0.3, 0.0, 0.0])
- Few adapters contribute
- Sharper selection

**High λ_sparsity**:
- Forces β toward one-hot
- May hurt performance if too aggressive
- Good for memory-constrained settings

**Trusting Sparsemax (threshold ≈ 0)**:
- Model learns which adapters are useful
- Natural sparsity from sparsemax activation
- More adaptive than arbitrary thresholds

## Running the Code

### Basic Training

```bash
python main.py --config=./exps/sdlora_c100.json
```

### Expected Output

```
[GumbelGate] Loaded pruning mask: 1 adapters already pruned
[Gumbel CL-LoRA] tau_init=5.0, tau_final=0.5, anneal_rate=0.999, lambda_sparsity=0.001
Task 2, Epoch 5/20 => Loss 0.152 (clf=0.148, sparse=4.844), tau=3.421, Train_accy 88.86

[Conditional Growth] Task 2 - Beta values: [0.9038 0.0000 0.0962]
[Conditional Growth] Pruning threshold: 1.0e-06 (sparsemax zeros)
✗ Pruned task 1 (beta=0.000000 ≈ 0)
[Conditional Growth] Active: 2/3 (66.7%)
[Conditional Growth] 🎯 Sublinear growth: 1 adapters pruned
```

**Note on sparsity loss**:
- Task 0: No sparsity loss (no previous adapters)
- Task 1: `sparse=0.0000` (only 1 previous adapter → deterministic)
- Task 2+: Non-zero sparsity loss (multiple adapters to choose from)

### Monitoring Beta Values

```python
import torch

# Load beta values
beta_data = torch.load('CF100/beta_values_task_3.pt')
print(f"Beta values: {beta_data['beta_values']}")
print(f"Pruning mask: {beta_data['pruning_mask']}")
print(f"Active tasks: {beta_data['pruning_mask'].sum().item()}")
```

## Key Implementation Details

### Why Sparsemax Instead of Softmax?

**Softmax**: Always produces non-zero probabilities
```
softmax([2, 1, 0.5, 0.1]) = [0.53, 0.19, 0.12, 0.08]  ← No zeros!
```

**Sparsemax**: Can produce exact zeros
```
sparsemax([2, 1, 0.5, 0.1]) = [0.70, 0.23, 0.07, 0.00]  ← True sparsity!
```

This enables **pruning**: When β_i = 0, we can safely discard adapter i.

### Granularity: Per-Task vs Per-Layer

Current implementation uses **per-task** granularity:
- One α_i and one β_i per task (shared across all LoRA layers)
- Simpler, fewer parameters
- Matches original SD-LoRA design

Alternative: **Per-layer** granularity would allow different selection per layer.

### Pruning Mask Persistence

Once an adapter is pruned (mask=0), it stays pruned:
```python
self.pruning_mask[task_id] = 0  # Permanent
torch.save(self.pruning_mask, mask_path)  # Saved to disk
```

On next task:
```python
if os.path.exists(mask_path):
    self.gumbel_gate.pruning_mask = torch.load(mask_path)  # Loaded
```

Pruned adapters:
- Still saved to disk (for reproducibility)
- But skipped in forward pass (lines 186-187)
- Enable sublinear growth

### Critical Bug Fixes

**NaN Issue (Fixed)**:
- **Root cause**: Pruned adapters had near-zero norms → division by zero in normalization
- **Solution**: Skip pruned adapters before computing normalized outputs (line 186)
- **Additional safeguard**: Added epsilon to all normalizations (1e-8)

**Mask Persistence Issue (Fixed)**:
- **Root cause**: Each task created new backbone → fresh mask (all 1s)
- **Solution**: Save mask after each task, load at initialization (lines 522-529)
- **Result**: Pruned adapters stay pruned across tasks

**Sparsemax Stability (Fixed)**:
- **Root cause**: -inf logits and division by zero in sparsemax
- **Solution**: Replace -inf with -1e9, add epsilon to divisions
- **Result**: Numerically stable even with extreme logit values

## Comparison with Original SD-LoRA

| Feature | Original SD-LoRA | Gumbel CL-LoRA |
|---------|------------------|----------------|
| Task selection | All tasks always active | Sparse selection via β |
| Magnitude learning | Fixed α=0.8 | Learnable α per task |
| Parameter growth | Linear (O(t)) | Sublinear (O(k) where k < t) |
| Gating mechanism | None | Gumbel-Sparsemax |
| Sparsity enforcement | None | Entropy regularization |
| Pruning | None | Adaptive (trust sparsemax zeros) |
| NaN handling | Potential issues | Multiple safeguards |

## Future Improvements

1. **Adaptive threshold**: Learn threshold based on task difficulty (though current approach trusts sparsemax)
2. **Top-k selection**: Replace threshold with fixed k active adapters
3. **Per-layer gating**: Different β for each LoRA layer (more flexibility)
4. **Hard selection**: Use `hard=True` in gumbel_sparsemax for evaluation
5. **Gradient clipping**: Additional stability for extreme scenarios

## References

- **Sparsemax**: Martins & Astudillo (2016) - "From Softmax to Sparsemax"
- **Gumbel-Softmax**: Jang et al. (2017) - "Categorical Reparameterization with Gumbel-Softmax"
- **SD-LoRA**: Wu et al. (2025) - "Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning"
