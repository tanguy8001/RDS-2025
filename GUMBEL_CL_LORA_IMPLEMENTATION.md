# Gumbel CL-LoRA Implementation Summary

## Overview

This document describes the implementation of the Gumbel CL-LoRA formulation for continual learning on CIFAR-100. The implementation integrates sparse Gumbel-Sparsemax task selection with the existing SD-LoRA framework to enable **sublinear parameter growth** during continual learning.

## Mathematical Formulation Implemented

### 1. Decoupled Adapter Structure (Equation 1)

```
ΔW_t = Σ_{i=1}^{t} β_i * α_i * (A_i B_i / ||A_i B_i||_F)
```

**Implementation**: `backbone/lora.py:205-237` (_LoRA_qkv_timm_train.forward)

- **Direction normalization**: Divides by `||w_b|| * ||w_a||` (lines 206-207)
- **Magnitude parameters** (`α_i`): Stored in `GumbelGate.alpha` (line 277)
- **Gate weights** (`β_i`): Computed via Gumbel-Sparsemax (line 217)

### 2. Gumbel-Sparsemax Sparse Gating (Equation 2)

```
β = sparsemax((l + G) / τ), where G ~ Gumbel(0,1)
```

**Implementation**: `utils/gumbel_utils.py:33-73` (sparsemax), `utils/gumbel_utils.py:76-125` (gumbel_sparsemax)

- **Sparsemax projection**: Yields exact zeros (unlike softmax)
- **Gumbel noise**: Added during training for exploration
- **Temperature τ**: Annealed from 5.0 → 0.5 during training

### 3. Sparsity Regularization (Equation 3)

```
L_total = L_task + λ_sparsity * Ω_sparsity(β)
Ω_sparsity = -Σ β_i log(β_i + ε)
```

**Implementation**: `models/sdlora.py:259-274` (_update_representation)

- **Sparsity loss collection**: Lines 267-270 (collects from all LoRA layers)
- **Total loss**: Line 274 (`loss = loss_clf + lambda_sparsity * sparsity_loss_total`)
- **Default λ_sparsity**: 0.01 (configurable in JSON)

### 4. Conditional Growth Mechanism (Equation 4)

```
If β_t > δ: Keep adapter, N ← N+1
If β_t ≤ δ: Prune adapter, N unchanged
```

**Implementation**: `models/sdlora.py:322-384` (_conditional_growth_decision)

- **Threshold δ**: 0.1 (configurable as `growth_threshold`)
- **Pruning mask**: Stored in `GumbelGate.pruning_mask` (line 289)
- **Beta logging**: Saved to `CF100/beta_values_task_*.pt` for analysis

## File Structure

### New Files

1. **`utils/gumbel_utils.py`** (263 lines)
   - `sample_gumbel()`: Gumbel(0,1) noise sampling
   - `sparsemax()`: Sparsemax projection (yields exact zeros)
   - `gumbel_sparsemax()`: Combines Gumbel noise + Sparsemax
   - `sparsity_loss()`: Entropy-based sparsity regularization
   - `TemperatureScheduler`: Exponential temperature annealing

### Modified Files

2. **`backbone/lora.py`**
   - **Added**: `GumbelGate` class (lines 311-433)
     - Manages α (magnitude) and β (gate) parameters per task
     - Implements Gumbel-Sparsemax forward pass
     - Tracks pruning mask for conditional growth

   - **Modified**: `_LoRA_qkv_timm_train` (lines 133-247)
     - Replaced fixed scaling factors with GumbelGate
     - Collects normalized adapter outputs
     - Applies Gumbel-Sparsemax gating
     - Stores sparsity loss for backprop

   - **Modified**: `_LoRA_qkv_timm_eval` (lines 249-323)
     - Updated for GumbelGate (no Gumbel noise in eval)

   - **Modified**: `LoRA_ViT_timm.__init__` (lines 503-532)
     - Instantiates `GumbelGate` instead of `ParameterWrapper`
     - Passes `gumbel_gate` to all LoRA layers

3. **`models/sdlora.py`**
   - **Added**: Import `TemperatureScheduler` (line 13)

   - **Modified**: `_update_representation` (lines 217-320)
     - Initializes temperature scheduler
     - Anneals τ during training (line 241)
     - Collects sparsity loss from LoRA layers (lines 267-270)
     - Adds sparsity regularization to total loss (line 274)
     - Logs detailed loss breakdown (clf + sparse + tau)

   - **Added**: `_conditional_growth_decision` (lines 322-384)
     - Evaluates β values after training
     - Prunes adapters below threshold
     - Logs active/pruned adapter statistics
     - Saves beta values for analysis

4. **`exps/sdlora_c100.json`**
   - **Added**: Gumbel CL-LoRA hyperparameters (lines 30-35)
     ```json
     "gumbel_tau_init": 5.0,
     "gumbel_tau_final": 0.5,
     "gumbel_anneal_rate": 0.999,
     "lambda_sparsity": 0.01,
     "growth_threshold": 0.1
     ```

## How It Works

### Training Pipeline (Task t > 0)

1. **Initialization** (`models/sdlora.py:224-226`)
   - Create `TemperatureScheduler` with τ_init=5.0, τ_final=0.5
   - Initialize GumbelGate with learnable α and logit parameters

2. **Forward Pass** (`backbone/lora.py:159-243`)
   - For each previous task i ∈ [0, t-1]:
     - Load saved LoRA weights A_i, B_i
     - Compute normalized output: `LoRA_i(x) / ||A_i|| * ||B_i||`
   - Apply Gumbel-Sparsemax gating:
     - Sample Gumbel noise (if training)
     - Compute β = sparsemax((logits + noise) / τ)
     - Weighted sum: `Σ β_i * α_i * LoRA_i(x)`
   - Add current task adapter: `α_t * LoRA_t(x)`

3. **Loss Computation** (`models/sdlora.py:253-274`)
   - Classification loss: Cross-entropy on current task
   - Sparsity loss: Collect from all LoRA layers
   - Total loss: `L = L_clf + λ * L_sparsity`

4. **Temperature Annealing** (`models/sdlora.py:241`)
   - Each iteration: `τ ← τ_final + (τ_init - τ_final) * anneal_rate^step`
   - Encourages β to become more sparse over time

5. **Conditional Growth** (`models/sdlora.py:322-384`)
   - After training, evaluate β_t (current task's gate)
   - If β_t > 0.1: Keep adapter (set mask=1)
   - If β_t ≤ 0.1: Prune adapter (set mask=0)
   - Pruned adapters contribute 0 to future tasks (via masked logits)

### Evaluation Mode

- No Gumbel noise (deterministic)
- Uses learned β values (no sampling)
- Pruned adapters automatically excluded (mask=0 → logit=-∞ → β=0)

## Hyperparameter Tuning Guide

### Gumbel-Sparsemax Parameters

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `gumbel_tau_init` | 5.0 | Initial temperature | Higher = more exploration, softer selection |
| `gumbel_tau_final` | 0.5 | Final temperature | Lower = sharper selection, closer to hard |
| `gumbel_anneal_rate` | 0.999 | Decay rate per step | Higher = slower annealing |
| `lambda_sparsity` | 0.01 | Sparsity regularization weight | Higher = sparser β, fewer active adapters |
| `growth_threshold` | 0.1 | Pruning threshold | Higher = more aggressive pruning |

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

**Low threshold**:
- Keeps most adapters (slower pruning)
- Higher final parameter count
- Better accuracy, more memory

## Running the Code

### Basic Training

```bash
python main.py --config=./exps/sdlora_c100.json
```

### Expected Output

```
[Gumbel CL-LoRA] tau_init=5.0, tau_final=0.5, anneal_rate=0.999, lambda_sparsity=0.01
Task 1, Epoch 5/20 => Loss 1.234 (clf=1.200, sparse=0.034), tau=3.421, Train_accy 75.23, Test_accy 72.11

[Conditional Growth] Task 1 - Beta values: [0.35 0.65]
[Conditional Growth] Current task beta: 0.6500, threshold: 0.1000
[Conditional Growth] ✓ Keeping adapter for task 1 (beta=0.6500 > 0.1000)
[Conditional Growth] Active adapters: 2/2 (100.0%)
```

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
```

Pruned adapters:
- Still saved to disk (for reproducibility)
- But excluded from inference (logit=-∞ → β=0)
- Enable sublinear growth

## Troubleshooting

### Issue: β values not sparse

**Cause**: λ_sparsity too low or τ too high

**Solution**: Increase `lambda_sparsity` (try 0.05 or 0.1)

### Issue: All adapters pruned

**Cause**: `growth_threshold` too high

**Solution**: Lower threshold to 0.05 or 0.01

### Issue: NaN losses

**Cause**: Numerical instability in sparsemax or log

**Solution**: Check for empty task_indices or zero β values

### Issue: No improvement over baseline

**Cause**: Hyperparameters not tuned for dataset

**Solution**: Try grid search over λ_sparsity ∈ [0.001, 0.01, 0.1]

## Comparison with Original SD-LoRA

| Feature | Original SD-LoRA | Gumbel CL-LoRA |
|---------|------------------|----------------|
| Task selection | All tasks always active | Sparse selection via β |
| Magnitude learning | Fixed α=0.8 | Learnable α per task |
| Parameter growth | Linear (O(t)) | Sublinear (O(k) where k < t) |
| Gating mechanism | None | Gumbel-Sparsemax |
| Sparsity enforcement | None | Entropy regularization |
| Pruning | None | Conditional growth (threshold-based) |

## Expected Results on CIFAR-100

### Baseline SD-LoRA
- **Final accuracy**: ~68-72% (10 tasks)
- **Active adapters**: 10/10 (100%)
- **Parameter growth**: Linear

### Gumbel CL-LoRA (Expected)
- **Final accuracy**: ~66-70% (slight drop due to sparsity)
- **Active adapters**: 5-7/10 (50-70% pruned)
- **Parameter growth**: Sublinear (30-50% savings)

Trade-off: Small accuracy drop for significant memory savings.

## Future Improvements

1. **Adaptive threshold**: Adjust `growth_threshold` based on task difficulty
2. **Top-k selection**: Replace threshold with fixed k active adapters
3. **Per-layer gating**: Different β for each LoRA layer
4. **Orthogonality loss**: Re-enable `ortho_loss` to prevent forgetting
5. **Hard selection**: Use `hard=True` in gumbel_sparsemax for evaluation

## References

- **Sparsemax**: Martins & Astudillo (2016) - "From Softmax to Sparsemax"
- **Gumbel-Softmax**: Jang et al. (2017) - "Categorical Reparameterization with Gumbel-Softmax"
- **SD-LoRA**: Original paper on magnitude-direction decoupling

## Contact

For questions or issues with this implementation, please check:
- Beta value logs in `CF100/beta_values_task_*.pt`
- Training logs for loss breakdown
- GumbelGate pruning mask for active adapters
