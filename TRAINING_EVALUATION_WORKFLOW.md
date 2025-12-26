# Training and Evaluation Workflow Documentation

## Overview

This document provides a detailed explanation of the training and evaluation workflows in the RDS-2025 Gumbel CL-LoRA implementation, with special focus on verifying that evaluation uses ONLY pruned (kept) adapters and not all trained adapters.

## Key Components

### 1. Model Hierarchy

```
Learner (models/sdlora.py)
  └─ _network: IncrementalNet (utils/inc_net.py)
      ├─ backbone: LoRA_ViT_timm (backbone/lora.py)
      │   ├─ lora_vit: Vision Transformer with LoRA layers
      │   │   └─ blocks[i].attn.qkv: _LoRA_qkv_timm_train OR _LoRA_qkv_timm_eval
      │   ├─ gumbel_gate: GumbelGate (handles α and β parameters)
      │   └─ w_As, w_Bs: Current task's trainable LoRA adapters
      └─ fc: Classifier head (Linear layer)
```

### 2. Adapter Types

- **Training Layers** (`_LoRA_qkv_timm_train`): Used during training
  - Loads previous tasks' adapters (with normalization)
  - Includes current task's adapter (without normalization)
  - Applies Gumbel-Sparsemax gating with Gumbel noise
  
- **Evaluation Layers** (`_LoRA_qkv_timm_eval`): Used during evaluation
  - Loads ONLY saved and non-pruned adapters
  - Uses deterministic gating (no Gumbel noise)
  - Respects pruning mask

## Training Workflow

### Initialization (Task t)

**File:** `models/sdlora.py:_train()` → `update_network()` → `backbone/lora.py:LoRA_ViT_timm.__init__()`

1. **Load Previous Adapters** (`backbone/lora.py:544-550`)
   ```python
   saved_lora_A, saved_lora_B = {}, {}
   for i in range(self.task_id + 1):
       if os.path.exists(lora_w_a_{i}.pt) and os.path.exists(lora_w_b_{i}.pt):
           saved_lora_A['saved_A_{i}'] = torch.load(...)
           saved_lora_B['saved_B_{i}'] = torch.load(...)
   ```

2. **Initialize GumbelGate** (`backbone/lora.py:552-569`)
   - Creates learnable α (magnitude) and l (logit) parameters for each task
   - Loads pruning mask from previous task if exists
   - Loads gate parameters (alphas, logits) from previous task
   ```python
   self.gumbel_gate = GumbelGate(max_tasks=20, init_alpha=0.8, init_logit=0.0)
   if os.path.exists(gumbel_gate_task_{task_id-1}.pt):
       state = torch.load(...)
       self.gumbel_gate.load_parameters(state['alphas'], state['gate_logits'], state['pruning_mask'])
   ```

3. **Create Training Layers** (`backbone/lora.py:590-594`)
   ```python
   blk.attn.qkv = _LoRA_qkv_timm_train(
       w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
       self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank, 
       self.gumbel_gate, tau=self.tau
   )
   ```

### Forward Pass During Training

**File:** `backbone/lora.py:_LoRA_qkv_timm_train.forward()`

1. **Load Previous Tasks** (lines 193-227)
   ```python
   for i in range(self.task_id):
       # ✓ CHECK PRUNING MASK FIRST
       if self.gumbel_gate.pruning_mask[i] == 0:
           continue  # Skip pruned adapters
       
       # Load saved A_i, B_i from disk
       # Compute: normalized_output = adapter(x) / (||A|| * ||B|| + ε)
       normalized_adapters_q.append(norm_q)
       normalized_adapters_v.append(norm_v)
   ```

2. **Add Current Task** (lines 229-236)
   ```python
   # Current task adapter (NOT normalized)
   norm_q_curr = self._compute_adapter_out(..., normalize=False)
   norm_v_curr = self._compute_adapter_out(..., normalize=False)
   normalized_adapters_q.append(norm_q_curr)
   normalized_adapters_v.append(norm_v_curr)
   ```

3. **Apply Gumbel-Sparsemax Gating** (lines 239-245)
   ```python
   delta_q, beta_q = self.gumbel_gate(
       normalized_adapters_q, task_indices, tau=self.tau, training=True
   )
   # beta_q = Sparsemax((l + Gumbel(0,1)) / tau)
   # delta_q = Σ β_i α_i × normalized_adapter_i
   ```

4. **Apply to QKV** (lines 256-258)
   ```python
   qkv = self.qkv(x)  # Pretrained weights
   qkv[:, :, :dim] += delta_q  # Add to Q
   qkv[:, :, -dim:] += delta_v  # Add to V
   ```

### Two-Phase Training

**File:** `models/sdlora.py:_update_representation()`

#### Phase 1: Learn Magnitude + Task Adapters (lines 235-272)
- **Trainable:** LoRA adapters (A, B), magnitude α, classifier fc
- **Frozen:** Gate logits l
- **Loss:** L_clf (cross-entropy)
- **Optimizer:** SGD with separate learning rates
  - LoRA: `lrate` (e.g., 0.001)
  - Alpha: `alpha_lr` (e.g., 0.01)
  - FC: `lrate`

#### Phase 2: Learn Selection (lines 274-295)
- **Trainable:** Gate logits l
- **Frozen:** LoRA adapters (A, B), magnitude α, classifier fc
- **Loss:** L_clf + λ * L_sparsity
  - Sparsity loss encourages sparse β values
  - `L_sparsity = -Σ β_i log(β_i + ε)`
- **Optimizer:** AdamW with `beta_lr`
- **Temperature Annealing:** τ decays from 5.0 → 0.5 to sharpen selection

### Conditional Growth Decision

**File:** `models/sdlora.py:_conditional_growth_decision()`

After training task t, evaluate final β values:

```python
with torch.no_grad():
    betas = gumbel_gate.get_betas(task_indices, tau=0.5)

current_beta = betas[-1].item()

if current_beta < 0.05:
    gumbel_gate.prune_task(current_task)  # Set mask[t] = 0
    logging.info("✗ PRUNED")
else:
    gumbel_gate.keep_task(current_task)   # Set mask[t] = 1
    logging.info("✓ KEPT")

# Save pruning mask and gate parameters
torch.save(gumbel_gate.pruning_mask, 'pruning_mask.pt')
torch.save({
    'alphas': [...],
    'gate_logits': [...],
    'pruning_mask': gumbel_gate.pruning_mask
}, 'gumbel_gate_task_{t}.pt')
```

**Key Point:** Once pruned (mask=0), the adapter stays pruned permanently across all future tasks.

### After Task

**File:** `models/sdlora.py:_train()`

Save trained adapters and classifier:

```python
self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
self._network.save_fc(save_lora_name, self._cur_task)
```

Saves:
- `lora_w_a_{task_id}.pt` - LoRA A matrices
- `lora_w_b_{task_id}.pt` - LoRA B matrices  
- `CLs_weight{task_id}.pt` - Classifier weights
- `CLs_bias{task_id}.pt` - Classifier bias
- `pruning_mask.pt` - Pruning mask
- `gumbel_gate_task_{task_id}.pt` - Gate parameters (α, l, mask)

## Evaluation Workflow

### Entry Point

**File:** `trainer.py:_train()` → `model.eval_task()` → `models/base.py:eval_task()`

```python
def eval_task(self):
    y_pred, y_true = self._eval_cnn(self.test_loader)  # Line 117
    cnn_accy = self._evaluate(y_pred, y_true)
    # ... NME evaluation if applicable ...
    return cnn_accy, nme_accy
```

### CNN Evaluation

**File:** `models/base.py:_eval_cnn()`

```python
def _eval_cnn(self, loader):
    self._network.eval()  # Set to eval mode
    y_pred, y_true = [], []
    for _, inputs, targets in loader:
        with torch.no_grad():
            # ✓ CRITICAL: eval=True is passed here
            outputs = self._network.forward(inputs, eval=True)['logits']  # Line 166
        predicts = torch.topk(outputs, k=self.topk, ...)
        y_pred.append(predicts.cpu().numpy())
        y_true.append(targets.cpu().numpy())
    return np.concatenate(y_pred), np.concatenate(y_true)
```

### IncrementalNet Forward with eval=True

**File:** `utils/inc_net.py:IncrementalNet.forward()`

```python
def forward(self, x, ortho_loss=False, eval=False):
    if eval:  # Line 303
        # ✓ Passes eval=True to backbone
        out = self.backbone(x, eval=True)  # Line 304
        out.update({"features": x})
        return out  # Returns backbone's output (includes logits)
    else:
        # Training path: uses self.fc classifier
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out
```

### Backbone Forward with eval=True

**File:** `backbone/lora.py:LoRA_ViT_timm.forward()`

```python
def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
    if eval:  # Line 691
        # ✓ Recreates model with EVAL layers
        self.reset(eval=True)  # Line 692
        return self.lora_vit(x)  # Returns logits from lora_vit.head
    else:
        return self.lora_vit(x)
```

### Model Reset for Evaluation

**File:** `backbone/lora.py:reset()`

```python
def reset(self, eval=False):
    # ✓ Reinitializes with eval=True to create eval layers
    self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)
```

**File:** `backbone/lora.py:LoRA_ViT_timm.__init__()` with `eval=True`

```python
# Create EVAL layers instead of training layers
for t_layer_i, blk in enumerate(vit_model.blocks):
    if not eval:
        blk.attn.qkv = _LoRA_qkv_timm_train(...)
    else:
        # ✓ Creates evaluation layers
        blk.attn.qkv = _LoRA_qkv_timm_eval(
            self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, 
            t_layer_i, self.rank, self.gumbel_gate, self.save_file
        )  # Line 596

# Set classifier head with saved weights
self.reset_lora_vit_head()  # Loads CLs_weight{task_id-1}.pt and CLs_bias{task_id-1}.pt
```

### Evaluation Forward Pass

**File:** `backbone/lora.py:_LoRA_qkv_timm_eval.forward()`

```python
def forward(self, x):
    normalized_adapters_q = []
    normalized_adapters_v = []
    task_indices = []
    
    # ✓ CRITICAL: Only loads tasks 0 to task_id-1 (completed tasks)
    for i in range(self.task_id):  # Line 301
        # ✓ CHECK 1: Skip pruned tasks
        if self.gumbel_gate.pruning_mask[i] == 0:  # Line 303
            continue
        
        # ✓ CHECK 2: Skip if not saved yet
        if 'saved_A_{i}' not in self.saved_A or 'saved_B_{i}' not in self.saved_B:
            continue
        
        task_indices.append(i)
        
        # Load saved adapters from disk
        saved_A_i = self.saved_A['saved_A_' + str(i)]
        saved_B_i = self.saved_B['saved_B_' + str(i)]
        
        # Extract Q and V adapters
        adapters = list(zip(saved_A_i, saved_B_i))
        A_q, B_q = adapters[self.t_layer_i * 2]
        A_v, B_v = adapters[self.t_layer_i * 2 + 1]
        
        # Create temporary layers (frozen)
        w_a_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_v = nn.Linear(self.rank, self.dim, bias=False)
        
        w_a_q.weight = nn.Parameter(A_q.weight.to(x.device), requires_grad=False)
        w_b_q.weight = nn.Parameter(B_q.weight.to(x.device), requires_grad=False)
        w_a_v.weight = nn.Parameter(A_v.weight.to(x.device), requires_grad=False)
        w_b_v.weight = nn.Parameter(B_v.weight.to(x.device), requires_grad=False)
        
        # ✓ Normalize all except the latest completed task
        is_latest = (i == self.task_id - 1)
        norm_q = self._compute_adapter_out(w_a_q, w_b_q, x, normalize=not is_latest)
        norm_v = self._compute_adapter_out(w_a_v, w_b_v, x, normalize=not is_latest)
        
        normalized_adapters_q.append(norm_q)
        normalized_adapters_v.append(norm_v)
    
    # ✓ Apply gating: NO Gumbel noise, deterministic, tau=0.5
    if len(normalized_adapters_q) > 0:
        delta_q, _ = self.gumbel_gate(
            normalized_adapters_q, task_indices, tau=0.5, training=False  # Line 342
        )
        delta_v, _ = self.gumbel_gate(
            normalized_adapters_v, task_indices, tau=0.5, training=False
        )
    else:
        delta_q, delta_v = 0, 0
    
    # Apply to QKV
    qkv = self.qkv(x)
    qkv[:, :, :self.dim] += delta_q
    qkv[:, :, -self.dim:] += delta_v
    return qkv
```

### Key Differences: Training vs Evaluation Layers

| Aspect | Training Layer | Evaluation Layer |
|--------|---------------|------------------|
| **Layer Type** | `_LoRA_qkv_timm_train` | `_LoRA_qkv_timm_eval` |
| **Tasks Loaded** | 0 to task_id (includes current) | 0 to task_id-1 (only completed) |
| **Current Task** | ✓ Included (trainable) | ✗ Not included |
| **Pruning Check** | ✓ Yes, skips pruned tasks | ✓ Yes, skips pruned tasks |
| **Gumbel Noise** | ✓ Yes (exploration) | ✗ No (deterministic) |
| **Temperature** | Variable (annealed) | Fixed (0.5) |
| **Normalization** | Previous: Yes, Current: No | All except latest: Yes |
| **Gradient** | Enabled for current task | Disabled (no_grad) |

## Verification: Does Evaluation Use ONLY Pruned Adapters?

### ✓ YES - Here's the Evidence:

1. **Eval Path is Used** (`models/base.py:166`)
   ```python
   outputs = self._network.forward(inputs, eval=True)['logits']
   ```

2. **Eval Layers are Created** (`backbone/lora.py:596`)
   ```python
   if not eval:
       blk.attn.qkv = _LoRA_qkv_timm_train(...)
   else:
       blk.attn.qkv = _LoRA_qkv_timm_eval(...)  # ✓ USES EVAL LAYERS
   ```

3. **Pruning Mask is Checked** (`backbone/lora.py:303`)
   ```python
   for i in range(self.task_id):
       if self.gumbel_gate.pruning_mask[i] == 0:
           continue  # ✓ SKIPS PRUNED ADAPTERS
   ```

4. **Current Task is NOT Included** (`backbone/lora.py:301`)
   ```python
   for i in range(self.task_id):  # ✓ Loops 0 to task_id-1, NOT task_id
   ```

5. **Mask is Persistent** (`backbone/lora.py:557-566`)
   ```python
   # Mask loaded at initialization from previous task
   if os.path.exists(mask_path):
       self.gumbel_gate.pruning_mask = torch.load(mask_path)
   ```

### Example Scenario:

**After Task 9 Training:**
- Tasks 0-9 trained
- Pruning decisions: Keep [0, 2], Prune [1, 3, 4, 5, 6, 7, 8, 9]
- Pruning mask: `[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]`
- Saved: `pruning_mask.pt`, `gumbel_gate_task_9.pt`

**During Evaluation (after Task 9):**
1. `self.task_id = 9` (next task to train is 10)
2. Loop: `for i in range(9)` → processes tasks 0-8
3. Task 0: mask=1 → ✓ Loaded
4. Task 1: mask=0 → ✗ Skipped (pruned)
5. Task 2: mask=1 → ✓ Loaded
6. Tasks 3-8: mask=0 → ✗ Skipped (pruned)
7. Task 9: NOT in loop → ✗ Not included
8. **Result:** Only tasks 0 and 2 are used for evaluation

## Potential Issues and Clarifications

### Issue 1: task_id Semantics

**Question:** What does `self.task_id` represent during evaluation?

**Answer:** `self.task_id` represents the NEXT task to be trained, NOT the current completed task.

- After completing task 0: `task_id = 1`
- After completing task 1: `task_id = 2`
- After completing task N: `task_id = N + 1`

This is why eval loops `for i in range(self.task_id)` correctly loads tasks 0 to N.

### Issue 2: Latest Task Normalization

**Question:** Why does eval normalize all except the latest task?

**Code:** `backbone/lora.py:332-334`
```python
is_latest = (i == self.task_id - 1)
norm_q = self._compute_adapter_out(w_a_q, w_b_q, x, normalize=not is_latest)
```

**Answer:** This matches training behavior where the current task adapter is not normalized. The latest completed task (task_id - 1) was not normalized during its training, so it shouldn't be normalized during evaluation to maintain consistency.

**Clarification:** This might be a design choice, but could introduce inconsistency. Consider normalizing all completed tasks uniformly.

### Issue 3: Classifier Head Consistency

**Question:** Does the backbone's lora_vit.head match IncrementalNet's fc layer?

**During Training:**
- Uses `IncrementalNet.fc` (updated by `update_fc()`)
- Forward: `x = backbone(x); out = fc(x)`

**During Evaluation:**
- Uses `backbone.lora_vit.head` (loaded by `reset_lora_vit_head()`)
- Forward: `out = backbone(x, eval=True)` → returns logits from lora_vit.head

**Verification Needed:** Ensure both classifiers have the same weights. The code saves fc to disk and loads it into lora_vit.head during reset, so they should match.

## Summary

### Training Instantiation Flow:
```
Learner.__init__()
  └─ IncrementalNet.__init__(args, pretrained=True)
      └─ get_backbone(args, pretrained=True)
          └─ LoRA_ViT_timm.__init__(eval=False, index=True)
              ├─ Load saved adapters (A, B) from previous tasks
              ├─ Load GumbelGate parameters (α, l, mask)
              ├─ Create trainable adapters for current task
              └─ Create _LoRA_qkv_timm_train layers
```

### Evaluation Instantiation Flow:
```
_eval_cnn(loader)
  └─ _network.forward(inputs, eval=True)
      └─ IncrementalNet.forward(x, eval=True)
          └─ backbone(x, eval=True)
              └─ LoRA_ViT_timm.forward(x, eval=True)
                  └─ reset(eval=True)
                      └─ LoRA_ViT_timm.__init__(eval=True, index=False)
                          ├─ Load saved adapters (A, B) from all completed tasks
                          ├─ Load GumbelGate parameters (α, l, mask)
                          ├─ Create _LoRA_qkv_timm_eval layers (NO trainable adapters)
                          └─ Load classifier weights into lora_vit.head
```

### Training Forward Flow:
```
model(inputs)
  └─ IncrementalNet.forward(inputs, eval=False)
      ├─ x = backbone(x)  # Uses _LoRA_qkv_timm_train layers
      │   ├─ Load previous non-pruned tasks (normalized)
      │   ├─ Add current task (not normalized)
      │   └─ Apply Gumbel-Sparsemax gating
      └─ out = fc(x)  # Classifier logits
```

### Evaluation Forward Flow:
```
model(inputs, eval=True)
  └─ IncrementalNet.forward(inputs, eval=True)
      └─ out = backbone(x, eval=True)  # Uses _LoRA_qkv_timm_eval layers
          ├─ Load ONLY completed non-pruned tasks (normalized)
          ├─ NO current task included
          ├─ Apply deterministic gating (no Gumbel noise)
          └─ Return logits from lora_vit.head
```

## Conclusion

**✓ CONFIRMED:** The evaluation workflow correctly uses ONLY pruned (kept) adapters.

**Key Mechanisms:**
1. Separate training and evaluation layer types ensure proper adapter selection
2. Pruning mask is checked during both training and evaluation forward passes
3. Current task is excluded from evaluation (only completed tasks 0 to task_id-1)
4. Eval path is properly triggered by `eval=True` parameter throughout the call stack

**Recommendations:**
1. Consider normalizing the latest completed task for consistency
2. Add assertions to verify pruning mask integrity
3. Document the task_id semantics clearly (next task to train, not current task)
4. Consider adding unit tests to verify pruning behavior
