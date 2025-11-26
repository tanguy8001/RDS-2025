# SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning

Welcome to the official code repository for [SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning **(ICLR 2025, Oral)**](https://openreview.net/pdf?id=5U1rlpX68A).

If you find this code useful in your research then please cite  
```bibtex
@inproceedings{
wu2025sdlora,
title={{SD}-Lo{RA}: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning},
author={Yichen Wu and Hongming Piao and Long-Kai Huang and Renzhen Wang and Wanhua Li and Hanspeter Pfister and Deyu Meng and Kede Ma and Ying Wei},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5U1rlpX68A}
}
``` 

## 👀 Introduction
![SD-LoRA](imgs/intro.jpg)

- SD-LoRA introduces a decoupled learning strategy for the magnitude and direction of LoRA components to achieve scalable continual learning without rehearsal of huge sample features.
- It demonstrates a strong stability-plasticity trade-off by converging to overlapping low-loss regions across sequential tasks, supported by empirical and theoretical analysis.
- SD-LoRA and its two variants enable end-to-end optimization and efficient inference without component selection, achieving state-of-the-art performance on multiple CL benchmarks and foundation models.

## 🔬 Gumbel CL-LoRA: Sparse Continual Learning

This repository includes **Gumbel CL-LoRA**, a clean implementation for **sublinear parameter growth** in continual learning via sparse task selection.

### **Mathematical Formulation**

**1. Decoupled Adapter Structure (Eq. 1)**
```
ΔW_t = Σ_{i=1}^{t} β_i α_i × (A_i B_i / (||A_i|| × ||B_i||))
```
- Normalize LoRA directions (memory-efficient approximation)
- Learn magnitudes (α) separately from directions
- Sparse gating weights (β) from Sparsemax

**2. Gumbel-Sparsemax Gating (Eq. 2)**
```
β = Sparsemax((l + G) / τ),  G ~ Gumbel(0,1)
```
- Sparsemax produces **exact zeros** (unlike softmax)
- Gumbel noise enables stochastic exploration
- Temperature τ anneals: 5.0 → 0.5

**3. Training Loss (Eq. 3)**
```
L = L_task + λ_sparsity × Ω_sparsity(β)
Ω_sparsity = -Σ β_i log(β_i + ε)
```
- Entropy penalty encourages sparse β
- λ_sparsity = 0.01 (default)

**4. Conditional Growth (Eq. 4)**
```
if β_t = 0:  PRUNE adapter
if β_t > 0:  KEEP adapter
```
- **Scale-invariant**: No arbitrary threshold needed
- Sparsemax naturally produces exact zeros
- Decision based on optimization geometry

### **Key Features**
✅ **Memory-efficient normalization**: Uses `||A|| × ||B||` (avoids large matrix products)
✅ **Unified gating**: ALL tasks (including current) go through same mechanism
✅ **Efficient**: Reuses layer structure from original SD-LoRA (no OOM issues)
✅ **Sparsity regularization**: Entropy penalty on β
✅ **Natural pruning**: β = 0 → prune (no arbitrary threshold)

### **Hyperparameters** (see `exps/sdlora_c100.json`)
```json
{
  "gumbel_tau_init": 5.0,        // Initial temp (exploration)
  "gumbel_tau_final": 0.5,       // Final temp (exploitation)
  "gumbel_anneal_rate": 0.999,   // Exponential decay rate
  "lambda_sparsity": 0.01        // Sparsity regularization weight
}
```

### **Implementation Highlights**
- `backbone/lora.py`: `GumbelGate` module with Sparsemax gating
- `utils/gumbel_utils.py`: Sparsemax, Gumbel sampling, sparsity loss
- `models/sdlora.py`: Training loop with temperature annealing + sparsity loss

## 📜 Results
![SD-LoRA](imgs/results1.jpg)
![SD-LoRA](imgs/results2.jpg)
- To run the experiments, download the datasets to /data/ and execute:
   ```bash
  bash run.sh
- For your convenience, we have provided the running logs in the log file, where you can find detailed performance results for all streaming tasks.



## 🙏 Acknowledgement
This repo is built upon the following projects:

* [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)
* [PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
