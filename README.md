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

## 🔬 Gumbel-Sparsemax Extension

This repository includes an experimental extension implementing **Gumbel CL-LoRA** for sublinear parameter growth during continual learning:

**Core Mechanism:**
- **Decoupled magnitude-direction learning**: LoRA directions are normalized (`||A_i B_i||_F^{-1} A_i B_i`), magnitudes learned via per-task α parameters
- **Gumbel-Sparsemax gating**: Sparse task selection using Sparsemax (exact zeros) instead of softmax, with Gumbel noise for stochastic exploration
- **Sparsity regularization**: Entropy loss `-Σ β_i log(β_i)` encourages sparse gating weights
- **Temperature annealing**: τ: 5.0 → 0.5 for gradual transition from exploration to exploitation

**Conditional Growth:**
- After each task, adapters with β ≈ 0 (threshold=1e-6) are permanently pruned
- Trusts Sparsemax's learned selection: when model sets β → 0, adapter is not useful
- Pruning mask persists across tasks to prevent "reappearing" adapters
- Enables sublinear growth: memory scales sub-linearly with number of tasks

**Key Hyperparameters** (see `exps/sdlora_c100.json`):
```json
{
  "gumbel_tau_init": 5.0,        // Initial temperature (soft selection)
  "gumbel_tau_final": 0.5,       // Final temperature (sharp selection)
  "gumbel_anneal_rate": 0.999,   // Exponential decay rate
  "lambda_sparsity": 0.001,      // Sparsity regularization weight
  "growth_threshold": 1e-6       // Pruning threshold (trust sparsemax)
}
```

See `GUMBEL_CL_LORA_IMPLEMENTATION.md` for implementation details.

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
