# Workflow Summary: Training vs Evaluation

This document provides a concise summary of the training and evaluation workflows, answering the question: **"Does evaluation use ONLY the pruned adapters, and not all trained adapters?"**

## Quick Answer

**✅ YES - Evaluation uses ONLY non-pruned (kept) adapters.**

The evaluation workflow correctly:
1. Uses separate evaluation layers (`_LoRA_qkv_timm_eval`)
2. Checks the pruning mask before loading each adapter
3. Skips adapters where `pruning_mask[i] == 0` (pruned)
4. Excludes the current task (loads only completed tasks 0 to task_id-1)

## Model Instantiation

### During Training (models/sdlora.py)

```python
# Task t initialization
def update_network(self, index=False):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model = LoRA_ViT_timm(
        vit_model=model.eval(), 
        r=rank, 
        num_classes=10, 
        index=index,
        increment=self.args['increment'],
        filepath=self.args['filepath'], 
        cur_task_index=self._cur_task  # Current task ID
    )
    return model
```

**What happens:**
1. Load saved LoRA adapters from tasks 0 to cur_task_index
2. Load GumbelGate parameters (α, l, pruning_mask) from previous task
3. Create trainable adapters for current task
4. Create `_LoRA_qkv_timm_train` layers for each attention block

### During Evaluation (backbone/lora.py)

```python
def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
    if eval:
        # Recreate model with EVAL layers
        self.reset(eval=True)  
        return self.lora_vit(x)
```

**What happens:**
1. `reset(eval=True)` calls `__init__` with `eval=True`
2. Load saved LoRA adapters from all completed tasks
3. Load GumbelGate parameters (including pruning_mask)
4. Create `_LoRA_qkv_timm_eval` layers (NOT train layers)
5. Load classifier weights into lora_vit.head

## Forward Pass Workflows

### Training Forward Pass

**File:** `backbone/lora.py:_LoRA_qkv_timm_train.forward()`

```python
def forward(self, x):
    normalized_adapters_q = []
    normalized_adapters_v = []
    task_indices = []
    
    # 1. Load PREVIOUS tasks (0 to task_id-1)
    for i in range(self.task_id):
        if self.gumbel_gate.pruning_mask[i] == 0:
            continue  # Skip pruned
        
        task_indices.append(i)
        # Load saved A_i, B_i
        # Compute: output = adapter(x) / (||A|| * ||B|| + ε)
        normalized_adapters_q.append(norm_q)
        normalized_adapters_v.append(norm_v)
    
    # 2. Add CURRENT task (task_id)
    task_indices.append(self.task_id)
    norm_q_curr = self._compute_adapter_out(..., normalize=False)
    norm_v_curr = self._compute_adapter_out(..., normalize=False)
    normalized_adapters_q.append(norm_q_curr)
    normalized_adapters_v.append(norm_v_curr)
    
    # 3. Apply Gumbel-Sparsemax gating
    delta_q, beta_q = self.gumbel_gate(
        normalized_adapters_q, task_indices, 
        tau=self.tau, training=True  # WITH Gumbel noise
    )
    delta_v, beta_v = self.gumbel_gate(
        normalized_adapters_v, task_indices, 
        tau=self.tau, training=True
    )
    
    # 4. Apply to QKV
    qkv = self.qkv(x)
    qkv[:, :, :dim] += delta_q
    qkv[:, :, -dim:] += delta_v
    return qkv
```

**Key Points:**
- Includes tasks 0 to task_id (current task included)
- Current task is NOT normalized
- Uses Gumbel noise for exploration
- Temperature τ annealed from 5.0 to 0.5

### Evaluation Forward Pass

**File:** `backbone/lora.py:_LoRA_qkv_timm_eval.forward()`

```python
def forward(self, x):
    normalized_adapters_q = []
    normalized_adapters_v = []
    task_indices = []
    
    # Load ONLY completed tasks (0 to task_id-1)
    # task_id represents NEXT task to train
    for i in range(self.task_id):
        # CHECK 1: Skip pruned adapters
        if self.gumbel_gate.pruning_mask[i] == 0:
            continue
        
        # CHECK 2: Skip if not saved yet
        if 'saved_A_{i}' not in self.saved_A:
            continue
        
        task_indices.append(i)
        # Load saved A_i, B_i
        # Compute normalized output
        normalized_adapters_q.append(norm_q)
        normalized_adapters_v.append(norm_v)
    
    # Verify only non-pruned adapters
    assert all(self.gumbel_gate.pruning_mask[i] == 1 for i in task_indices)
    
    # Apply gating: NO Gumbel noise, deterministic
    if len(normalized_adapters_q) > 0:
        delta_q, _ = self.gumbel_gate(
            normalized_adapters_q, task_indices, 
            tau=0.5, training=False  # NO Gumbel noise
        )
        delta_v, _ = self.gumbel_gate(
            normalized_adapters_v, task_indices, 
            tau=0.5, training=False
        )
    else:
        delta_q, delta_v = 0, 0
    
    # Apply to QKV
    qkv = self.qkv(x)
    qkv[:, :, :dim] += delta_q
    qkv[:, :, -dim:] += delta_v
    return qkv
```

**Key Points:**
- Includes ONLY tasks 0 to task_id-1 (current task excluded)
- Skips pruned adapters (pruning_mask[i] == 0)
- No Gumbel noise (deterministic)
- Fixed temperature τ = 0.5

## Comparison Table

| Aspect | Training | Evaluation |
|--------|----------|------------|
| **Layer Type** | `_LoRA_qkv_timm_train` | `_LoRA_qkv_timm_eval` |
| **Tasks Loaded** | 0 to task_id | 0 to task_id-1 |
| **Current Task** | ✓ Included | ✗ Excluded |
| **Pruning Check** | ✓ Yes | ✓ Yes |
| **Gumbel Noise** | ✓ Yes | ✗ No |
| **Temperature** | Variable (5.0→0.5) | Fixed (0.5) |
| **Normalization** | Prev: Yes, Curr: No | All: Yes (except latest) |
| **Gradient** | Enabled for current | Disabled (no_grad) |

## Example Scenario

**After Training Task 9:**
- task_id = 10 (next task to train is 10)
- Pruning mask: `[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]`
  - Tasks 0, 2: Kept (mask=1)
  - Tasks 1, 3-9: Pruned (mask=0)

**Training Task 10:**
```python
for i in range(10):  # 0 to 9
    if pruning_mask[i] == 0: continue
    # Loads: 0, 2

# Add current task 10
# Active adapters: [0, 2, 10]
```

**Evaluation After Task 9:**
```python
for i in range(10):  # 0 to 9
    if pruning_mask[i] == 0: continue
    # Loads: 0, 2

# Current task 10 NOT included
# Active adapters: [0, 2]  ← ONLY 2 adapters!
```

## Call Stack

### Training
```
Learner.incremental_train()
  └─ self._train()
      └─ self._update_representation()
          └─ logits = self._network(inputs, ortho_loss=True)[0]['logits']
              └─ IncrementalNet.forward(x, ortho_loss=True)
                  └─ x = self.backbone(x, loss=True)
                      └─ LoRA_ViT_timm.forward(x, loss=True)
                          └─ return self.lora_vit(x), loss
                              └─ _LoRA_qkv_timm_train.forward(x)
                                  ├─ Load tasks 0 to task_id-1 (if not pruned)
                                  ├─ Add current task (task_id)
                                  └─ Apply Gumbel-Sparsemax gating
```

### Evaluation
```
Learner.eval_task()
  └─ self._eval_cnn(loader)
      └─ outputs = self._network.forward(inputs, eval=True)['logits']
          └─ IncrementalNet.forward(x, eval=True)
              └─ out = self.backbone(x, eval=True)
                  └─ LoRA_ViT_timm.forward(x, eval=True)
                      └─ self.reset(eval=True)
                          └─ __init__(eval=True)
                              └─ Create _LoRA_qkv_timm_eval layers
                      └─ return self.lora_vit(x)
                          └─ _LoRA_qkv_timm_eval.forward(x)
                              ├─ Load tasks 0 to task_id-1 ONLY (if not pruned)
                              ├─ Assertion: verify no pruned tasks
                              └─ Apply deterministic gating (no Gumbel noise)
```

## Verification

Run the verification script to see the behavior:

```bash
python verify_eval_workflow.py
```

**Output:**
```
Scenario: task_id = 10, mask = [1,0,1,0,0,0,0,0,0,0]
  Task 0: ✓ LOADED (pruning_mask[0] = 1)
  Task 1: SKIPPED (pruning_mask[1] = 0)
  Task 2: ✓ LOADED (pruning_mask[2] = 1)
  Tasks 3-9: SKIPPED (pruned)
  
Result:
  Active adapters: [0, 2]
  Count: 2/10 adapters (8 pruned)
  Sublinear growth: YES
  ✓ ASSERTION PASSED: All active adapters are non-pruned
```

## Conclusion

The implementation is **CORRECT**. Evaluation uses:
- ✅ ONLY non-pruned adapters (pruning_mask[i] == 1)
- ✅ ONLY completed tasks (0 to task_id-1, NOT task_id)
- ✅ Separate evaluation layers with deterministic gating
- ✅ No Gumbel noise during evaluation

This ensures that pruning decisions made during training are properly respected during evaluation, enabling sublinear parameter growth as intended by the Gumbel CL-LoRA design.

For detailed documentation, see [TRAINING_EVALUATION_WORKFLOW.md](TRAINING_EVALUATION_WORKFLOW.md).
