# Two-Phase Training Implementation

## Problem
Joint learning of selection parameters (β via logits) and magnitude parameters (α) creates an identifiability problem: if β_i shrinks, α_i can grow to compensate, defeating the sparse selection mechanism.

## Solution
Two-phase training at each task separates selection learning from magnitude calibration.

### Phase 1: Learn Selection (70% of epochs)
- **Freeze**: α parameters (fixed at 1.0)
- **Train**: Gate logits (l) → learns β via Gumbel-Sparsemax
- **Loss**: Classification + Sparsity regularization
- **Goal**: Learn *which* adapters to select based on task similarity

### Phase 2: Learn Magnitudes (30% of epochs)
- **Freeze**: Gate logits (l) → fixes β selection
- **Train**: α parameters
- **Loss**: Classification only (no sparsity)
- **Goal**: Learn *how much* to scale each selected adapter

## Pruning Decision
Uses **multiple Gumbel sampling** (10 samples) for robust decisions:
- Computes selection frequency: fraction of samples where adapter is selected (β > 0)
- **Prune** if `selection_freq < 0.2` (rarely selected across samples)
- **Keep** otherwise (robustly selected)

## Configuration (`exps/sdlora_c100.json`)
```json
{
  "gumbel_tau_init": 5.0,        // Initial temperature
  "gumbel_tau_final": 0.5,       // Final temperature
  "gumbel_anneal_rate": 0.999,   // Annealing rate per batch
  "lambda_sparsity": 0.1,        // Sparsity loss weight (Phase 1 only)
  "prune_threshold": 0.2,        // Prune if freq < 0.2
  "alpha_lr": 0.04               // Phase 2 learning rate (5x base lr)
}
```

**Why higher LR for Phase 2?**
- Alphas are scalar multipliers (not weight matrices)
- Need stronger gradient signal to move from initialization
- Fresh optimizer in Phase 2 avoids momentum contamination from Phase 1

## Implementation Details

### New Methods in `GumbelGate` (backbone/lora.py)
- `freeze_alphas(task_indices)`: Freeze α for Phase 1
- `unfreeze_alphas(task_indices)`: Unfreeze α for Phase 2
- `freeze_logits(task_indices)`: Freeze logits for Phase 2
- `unfreeze_logits(task_indices)`: Unfreeze logits for Phase 1
- `get_selection_frequency(task_indices, tau, num_samples)`: Multiple sampling for robust pruning

### Training Flow (models/sdlora.py)
1. `_update_representation()`: Orchestrates two-phase training
2. `_train_phase(phase=1 or 2)`: Single phase training loop
3. `_conditional_growth_decision()`: Multiple sampling-based pruning

## Benefits
1. **No compensation**: α and β learned separately, no identifiability issue
2. **Robust pruning**: Multiple samples reduce noise in selection decisions
3. **Clear semantics**: Phase 1 = what to select, Phase 2 = how much to scale
4. **Simple & clean**: Minimal code changes, easy to understand

## Usage
No changes needed - just run your existing training script. The two-phase training is automatic.
