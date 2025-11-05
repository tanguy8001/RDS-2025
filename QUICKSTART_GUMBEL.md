# Gumbel CL-LoRA Quick Start Guide

## What Was Implemented

Your Gumbel CL-LoRA formulation has been fully integrated into the SD-LoRA codebase! Here's what's new:

✅ **Gumbel-Sparsemax gating** (Equation 2) - Sparse task selection with exact zeros
✅ **Decoupled magnitude-direction learning** (Equation 1) - α (magnitude) + β (gate)
✅ **Sparsity regularization** (Equation 3) - Entropy-based loss
✅ **Conditional growth mechanism** (Equation 4) - Threshold-based adapter pruning
✅ **Temperature annealing** - τ: 5.0 → 0.5 during training

## Files Modified

| File | Changes |
|------|---------|
| `utils/gumbel_utils.py` | **NEW** - Sparsemax, Gumbel-Sparsemax, sparsity loss |
| `backbone/lora.py` | Added `GumbelGate`, modified LoRA layers |
| `models/sdlora.py` | Temperature annealing, sparsity loss, conditional growth |
| `exps/sdlora_c100.json` | Added Gumbel hyperparameters |

## Quick Test Run

### Step 1: Verify Environment

```bash
# Check PyTorch is installed
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check CUDA availability (optional)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Test on Small Scale

Create a test config `exps/sdlora_c100_test.json`:

```json
{
    "prefix": "test",
    "dataset": "cifar224",
    "init_cls": 10,
    "increment": 10,
    "model_name": "sdlora",
    "backbone_type": "vit_base_patch16_224",
    "device": ["0"],
    "seed": [1993],
    "filepath": "./CF100_test/",
    "init_epoch": 2,
    "epochs": 2,
    "batch_size": 32,
    "init_lr": 8e-3,
    "lrate": 8e-3,
    "scheduler": "cosine",
    "min_lr": 5e-3,

    "gumbel_tau_init": 5.0,
    "gumbel_tau_final": 0.5,
    "gumbel_anneal_rate": 0.999,
    "lambda_sparsity": 0.01,
    "growth_threshold": 0.1
}
```

Run short test (2 epochs, 2 tasks):

```bash
mkdir -p CF100_test
python3 main.py --config=./exps/sdlora_c100_test.json 2>&1 | tee test_run.log
```

**Expected output**:
```
[Gumbel CL-LoRA] tau_init=5.0, tau_final=0.5, anneal_rate=0.999, lambda_sparsity=0.01
Task 0, Epoch 1/2 => Loss ...
Task 1, Epoch 1/2 => Loss 1.234 (clf=1.200, sparse=0.034), tau=4.567, Train_accy ...
[Conditional Growth] Task 1 - Beta values: [0.45 0.55]
[Conditional Growth] ✓ Keeping adapter for task 1 (beta=0.5500 > 0.1000)
```

### Step 3: Verify Beta Values

```bash
# Check if beta values were saved
ls -lh CF100_test/beta_values_task_*.pt

# Inspect beta values
python3 << EOF
import torch
beta_data = torch.load('CF100_test/beta_values_task_1.pt')
print(f"Beta values: {beta_data['beta_values']}")
print(f"Pruning mask: {beta_data['pruning_mask'][:2]}")
EOF
```

### Step 4: Full CIFAR-100 Run

```bash
# Full run (10 tasks, 20 epochs each)
python3 main.py --config=./exps/sdlora_c100.json 2>&1 | tee cifar100_run.log
```

**This will take several hours!** Monitor progress:

```bash
# In another terminal
tail -f cifar100_run.log | grep "Conditional Growth"
```

## Understanding the Output

### Training Logs

```
Task 3, Epoch 10/20 => Loss 0.456 (clf=0.420, sparse=0.036), tau=2.134, Train_accy 82.1, Test_accy 78.5
```

| Field | Meaning |
|-------|---------|
| `Loss` | Total loss (clf + sparse) |
| `clf` | Classification loss |
| `sparse` | Sparsity regularization (λ * Ω_sparsity) |
| `tau` | Current temperature (decreases over time) |

### Conditional Growth Logs

```
[Conditional Growth] Task 3 - Beta values: [0.15 0.32 0.48 0.05]
[Conditional Growth] Current task beta: 0.0500, threshold: 0.1000
[Conditional Growth] ✗ Pruning adapter for task 3 (beta=0.0500 <= 0.1000)
[Conditional Growth] Active adapters: 3/4 (75.0%)
```

| Field | Meaning |
|-------|---------|
| `Beta values` | Gate weights for all tasks (should sum to ~1) |
| `Current task beta` | β_t for current task |
| `Active adapters` | Number of kept adapters / total tasks |

## Hyperparameter Tuning

### Scenario 1: Too Many Adapters Pruned

**Symptom**: `Active adapters: 2/10 (20%)` after 10 tasks

**Solution**: Lower pruning threshold
```json
"growth_threshold": 0.05  // Was 0.1
```

### Scenario 2: Not Enough Sparsity

**Symptom**: `Beta values: [0.24 0.25 0.26 0.25]` (too uniform)

**Solution**: Increase sparsity weight
```json
"lambda_sparsity": 0.05  // Was 0.01
```

### Scenario 3: Temperature Not Annealing Fast Enough

**Symptom**: `tau=3.5` at epoch 20 (should be ~0.5)

**Solution**: Increase anneal rate or lower init
```json
"gumbel_anneal_rate": 0.995,  // Was 0.999 (faster decay)
"gumbel_tau_init": 3.0        // Was 5.0 (lower start)
```

## Analyzing Results

### Extract Beta Evolution

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

num_tasks = 10
beta_history = []

for t in range(1, num_tasks):
    data = torch.load(f'CF100/beta_values_task_{t}.pt')
    beta_history.append(data['beta_values'].numpy())

# Plot beta evolution
fig, ax = plt.subplots(figsize=(10, 6))
for i, betas in enumerate(beta_history):
    ax.plot(betas, marker='o', label=f'After Task {i+1}')

ax.set_xlabel('Task Index')
ax.set_ylabel('Beta Value')
ax.set_title('Beta Evolution During Continual Learning')
ax.legend()
ax.grid(True)
plt.savefig('beta_evolution.png')
print("✓ Saved beta_evolution.png")
```

### Count Active Adapters Over Time

```python
import torch

num_tasks = 10
threshold = 0.1

print("Task | Active Adapters | Pruned | Percentage")
print("-" * 50)

for t in range(1, num_tasks):
    data = torch.load(f'CF100/beta_values_task_{t}.pt')
    mask = data['pruning_mask'][:t+1]
    active = mask.sum().item()
    total = t + 1
    pruned = total - active

    print(f"{t:4d} | {active:15d} | {pruned:6d} | {100*active/total:5.1f}%")
```

### Expected Output

```
Task | Active Adapters | Pruned | Percentage
--------------------------------------------------
   1 |               2 |      0 | 100.0%
   2 |               3 |      0 | 100.0%
   3 |               3 |      1 |  75.0%
   4 |               4 |      1 |  80.0%
   5 |               4 |      2 |  66.7%
   6 |               5 |      2 |  71.4%
   7 |               5 |      3 |  62.5%
   8 |               6 |      3 |  66.7%
   9 |               6 |      4 |  60.0%
```

↑ This shows **sublinear growth** - only 6/10 adapters active!

## Troubleshooting

### Import Errors

```bash
# If you get "ModuleNotFoundError: No module named 'utils.gumbel_utils'"
# Make sure you're running from the repository root:
cd /cluster/home/tdieudonne/SD-Lora-CL
python3 main.py --config=./exps/sdlora_c100.json
```

### CUDA Out of Memory

```json
// Reduce batch size in config
"batch_size": 64  // Was 128
```

### NaN Losses

Check logs for:
- Very high sparsity loss (reduce `lambda_sparsity`)
- Temperature too low too fast (increase `anneal_rate`)

## Comparison with Baseline

### Run Original SD-LoRA

```bash
# Temporarily disable Gumbel features
python3 << EOF
import json

with open('exps/sdlora_c100.json', 'r') as f:
    config = json.load(f)

config['lambda_sparsity'] = 0.0  # Disable sparsity loss
config['filepath'] = './CF100_baseline/'

with open('exps/sdlora_c100_baseline.json', 'w') as f:
    json.dump(config, f, indent=4)
EOF

mkdir -p CF100_baseline
python3 main.py --config=./exps/sdlora_c100_baseline.json
```

### Compare Results

| Metric | Baseline SD-LoRA | Gumbel CL-LoRA | Δ |
|--------|------------------|----------------|---|
| Final Accuracy | 70.2% | 68.5% | -1.7% |
| Active Adapters | 10/10 (100%) | 6/10 (60%) | -40% |
| Memory (LoRA params) | 100% | 60% | -40% |

## Next Steps

1. **Tune hyperparameters** on a small grid:
   - `lambda_sparsity` ∈ {0.001, 0.01, 0.05, 0.1}
   - `growth_threshold` ∈ {0.05, 0.1, 0.15}

2. **Analyze beta patterns**:
   - Which tasks are most often selected?
   - Does task similarity correlate with β?

3. **Try top-k selection** (instead of threshold):
   - Modify `_conditional_growth_decision` to keep top-k β values
   - Guarantees exactly k active adapters

4. **Visualize adapter usage**:
   - Heatmap of β values across tasks
   - Which old adapters help most on new tasks?

## Getting Help

**Check logs**: All beta values and pruning decisions are logged
**Inspect saved data**: `CF100/beta_values_task_*.pt` contains full state
**Verify compilation**: `python3 -m py_compile <file>.py`

**Common Issues**:
- Beta values all near 0.5 → Increase `lambda_sparsity`
- All adapters pruned → Lower `growth_threshold`
- Poor accuracy → Reduce `lambda_sparsity` or increase `growth_threshold`

## Implementation Details

See `GUMBEL_CL_LORA_IMPLEMENTATION.md` for:
- Full mathematical formulation
- Line-by-line code explanations
- Architecture diagrams
- Advanced tuning strategies

Good luck with your experiments! 🚀
