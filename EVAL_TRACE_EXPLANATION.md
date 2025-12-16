 # Evaluation Code Trace Explanation

## What is `_network`?

`_network` is an `IncrementalNet` object created in `models/sdlora.py`:
```python
# Line 26 in sdlora.py
self._network = IncrementalNet(args, True)
```

`IncrementalNet` is defined in `utils/inc_net.py` and contains:
- `self.backbone`: A `LoRA_ViT_timm` object (the Vision Transformer with LoRA adapters)
- `self.fc`: The classifier head

## Code Trace: WITHOUT `eval=True` (CURRENT BUGGY BEHAVIOR)

### Step 1: `models/base.py` line 166
```python
outputs = self._network.forward(inputs)['logits']
```
- Calls `IncrementalNet.forward()` with NO `eval` parameter

### Step 2: `utils/inc_net.py` line 302-322
```python
def forward(self, x, ortho_loss=False, eval=False):
    if eval:  # ❌ eval=False, so this is SKIPPED
        out = self.backbone(x, eval=True)
        ...
    else:  # ✅ THIS PATH IS TAKEN
        ...
        x = self.backbone(x)  # ❌ Calls backbone WITHOUT eval=True
        out = self.fc(x)
        ...
```

### Step 3: `backbone/lora.py` line 682-690
```python
def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
    if eval:  # ❌ eval=False, so this is SKIPPED
        self.reset(eval=True)
        return self.lora_vit(x)
    else:  # ✅ THIS PATH IS TAKEN
        return self.lora_vit(x)  # ❌ Uses TRAINING model directly
```

### Step 4: `backbone/lora.py` line 193-274 (`_LoRA_qkv_timm_train.forward()`)
This is the **TRAINING** layer that is used:
```python
def forward(self, x):
    # Load previous tasks (checks pruning mask) ✅
    for i in range(self.task_id):
        if self.gumbel_gate.pruning_mask[i] == 0:
            continue  # Skip pruned tasks
        # ... load adapter ...
    
    # === Add current task (trainable) ===
    task_indices.append(self.task_id)  # ❌ ALWAYS ADDS CURRENT TASK!
    
    norm_q_curr = self._compute_normalized_adapter(self.linear_a_q, self.linear_b_q, x)
    norm_v_curr = self._compute_normalized_adapter(self.linear_a_v, self.linear_b_v, x)
    
    normalized_adapters_q.append(norm_q_curr)  # ❌ CURRENT TASK ALWAYS INCLUDED!
    normalized_adapters_v.append(norm_v_curr)
```

**PROBLEM**: The training layer ALWAYS includes the current task adapter (line 243-250), 
even if it was just pruned! It doesn't check `pruning_mask[self.task_id]`.

---

## Code Trace: WITH `eval=True` (CORRECT BEHAVIOR)

### Step 1: `models/base.py` line 166 (FIXED)
```python
outputs = self._network.forward(inputs, eval=True)['logits']
```
- Calls `IncrementalNet.forward()` WITH `eval=True`

### Step 2: `utils/inc_net.py` line 302-306
```python
def forward(self, x, ortho_loss=False, eval=False):
    if eval:  # ✅ eval=True, so THIS PATH IS TAKEN
        out = self.backbone(x, eval=True)  # ✅ Passes eval=True to backbone
        out.update({"features": x})
        return out
```

### Step 3: `backbone/lora.py` line 682-685
```python
def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
    if eval:  # ✅ eval=True, so THIS PATH IS TAKEN
        self.reset(eval=True)  # ✅ Recreates model with EVAL layers
        return self.lora_vit(x)
```

### Step 4: `backbone/lora.py` line 610-611
```python
def reset(self, eval=False):
    self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)
```
- Reinitializes the model with `eval=True`

### Step 5: `backbone/lora.py` line 576-588
```python
if not eval:  # ❌ eval=True, so this is SKIPPED
    blk.attn.qkv = _LoRA_qkv_timm_train(...)
else:  # ✅ THIS PATH IS TAKEN
    # ... logging ...
    blk.attn.qkv = _LoRA_qkv_timm_eval(...)  # ✅ Creates EVAL layer
```

### Step 6: `backbone/lora.py` line 304-359 (`_LoRA_qkv_timm_eval.forward()`)
This is the **EVALUATION** layer that respects pruning:
```python
def forward(self, x):
    # Load all non-pruned tasks
    for i in range(self.task_id):  # ⚠️ Only loads previous tasks
        if self.gumbel_gate.pruning_mask[i] == 0:
            continue  # ✅ Skips pruned tasks
        # ... load adapter ...
    
    # ❌ CURRENT TASK NOT INCLUDED (but should check if not pruned)
```

**NOTE**: The eval layer only loads tasks 0 to task_id-1, but doesn't include the current task.
However, if the current task was pruned, it shouldn't be included anyway.

---

## The Real Bug

The issue is that **without `eval=True`**, the evaluation uses the **training layers** which:
1. Always include the current task adapter (line 243-250 in `_LoRA_qkv_timm_train`)
2. Don't check if the current task was pruned

So even though tasks 1, 3, 4, 5, 6, 7, 8, 9 are pruned, the evaluation still uses:
- Tasks 0, 2 (from previous tasks, correctly checked)
- Task 9 (current task, ALWAYS included regardless of pruning mask)

This is why you see good results despite most adapters being pruned!

---

## The Fix

Change `models/base.py` line 166 from:
```python
outputs = self._network.forward(inputs)['logits']
```

To:
```python
outputs = self._network.forward(inputs, eval=True)['logits']
```

This ensures evaluation uses the eval layers that properly respect the pruning mask.
