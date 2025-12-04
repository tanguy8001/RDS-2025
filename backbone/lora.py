# Sheng Wang at Feb 22 2023

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy
from utils.gumbel_utils import gumbel_sparsemax

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.
    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class _LoRA_qkv_timm(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x)) #* self.scaling_factor
        new_v = self.linear_b_v(self.linear_a_v(x)) #* self.scaling_factor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    
class _LoRA_qkv_timm_train(nn.Module):
    """
    Training-mode LoRA layer with Gumbel-Sparsemax gating.

    Implements Equation 1: ΔW_t = Σ β_i α_i × (A_i B_i / ||A_i B_i||_F)

    Key design:
    - ALL tasks (including current) go through the same gating mechanism
    - Normalization by Frobenius norm: ||AB||_F = sqrt(sum((A @ B)^2))
    - Previous tasks: frozen adapters loaded from disk
    - Current task: trainable adapters with learnable α, l
    """
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v,
                 task_id, saved_A, saved_B, t_layer_i, rank, gumbel_gate, tau=1.0):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = rank
        self.task_id = task_id
        self.t_layer_i = t_layer_i

        # Current task adapters (trainable)
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        # Previous task adapters (frozen, loaded from disk)
        self.saved_A = saved_A
        self.saved_B = saved_B

        # Gumbel gate for task selection
        self.gumbel_gate = gumbel_gate
        self.tau = tau

    def _compute_normalized_adapter(self, w_a, w_b, x):
        """
        Compute normalized adapter output: (B @ A @ x) / norm

        Note: Using ||B|| * ||A|| instead of ||BA||_F for memory efficiency.
        This is the same as original SD-LoRA (trade correctness for efficiency).

        Args:
            w_a: Low-rank matrix A [rank, dim]
            w_b: Low-rank matrix B [dim, rank]
            x: Input [batch, seq, dim]

        Returns:
            Normalized adapter output [batch, seq, dim]
        """
        # Compute adapter output: B(A(x))
        adapter_out = w_b(w_a(x))  # [B, seq, dim]

        # Memory-efficient normalization (avoids matrix multiplication)
        # ||B|| * ||A|| ≈ ||BA||_F (approximate, but avoids [dim, dim] intermediate)
        norm = torch.norm(w_b.weight) * torch.norm(w_a.weight) + 1e-8

        # Normalize
        return adapter_out / norm

    def forward(self, x):
        """
        Forward pass: ΔW = Σ β_i α_i × normalized_adapter_i

        Args:
            x: Input [batch, seq, dim]

        Returns:
            qkv: Output [batch, seq, 3*dim]
        """
        normalized_adapters_q = []
        normalized_adapters_v = []
        task_indices = []

        # === Load previous tasks (frozen) ===
        for i in range(self.task_id):
            
            if self.gumbel_gate.pruning_mask[i] == 0:
                continue

            task_indices.append(i)

            # Load saved adapters
            saved_A_i = self.saved_A['saved_A_' + str(i)]
            saved_B_i = self.saved_B['saved_B_' + str(i)]

            # Extract Q and V adapters for this layer
            adapters = list(zip(saved_A_i, saved_B_i))
            A_q, B_q = adapters[self.t_layer_i * 2]
            A_v, B_v = adapters[self.t_layer_i * 2 + 1]

            # Create temporary linear layers (no gradient)
            w_a_q = nn.Linear(self.dim, self.rank, bias=False)
            w_b_q = nn.Linear(self.rank, self.dim, bias=False)
            w_a_v = nn.Linear(self.dim, self.rank, bias=False)
            w_b_v = nn.Linear(self.rank, self.dim, bias=False)

            # Load weights (frozen)
            w_a_q.weight = nn.Parameter(A_q.weight.to(x.device), requires_grad=False)
            w_b_q.weight = nn.Parameter(B_q.weight.to(x.device), requires_grad=False)
            w_a_v.weight = nn.Parameter(A_v.weight.to(x.device), requires_grad=False)
            w_b_v.weight = nn.Parameter(B_v.weight.to(x.device), requires_grad=False)

            # Compute normalized outputs
            norm_q = self._compute_normalized_adapter(w_a_q, w_b_q, x)
            norm_v = self._compute_normalized_adapter(w_a_v, w_b_v, x)

            normalized_adapters_q.append(norm_q)
            normalized_adapters_v.append(norm_v)

        # === Add current task (trainable) ===
        task_indices.append(self.task_id)

        norm_q_curr = self._compute_normalized_adapter(self.linear_a_q, self.linear_b_q, x)
        norm_v_curr = self._compute_normalized_adapter(self.linear_a_v, self.linear_b_v, x)

        normalized_adapters_q.append(norm_q_curr)
        normalized_adapters_v.append(norm_v_curr)

        # === Apply Gumbel-Sparsemax gating ===
        if len(normalized_adapters_q) > 0:
            delta_q, beta_q = self.gumbel_gate(
                normalized_adapters_q, task_indices, tau=self.tau, training=self.training
            )
            delta_v, beta_v = self.gumbel_gate(
                normalized_adapters_v, task_indices, tau=self.tau, training=self.training
            )

            # Store beta for sparsity loss computation
            self.last_beta_q = beta_q
            self.last_beta_v = beta_v
        else:
            delta_q, delta_v = 0, 0
            self.last_beta_q = None
            self.last_beta_v = None

        # === Apply to QKV ===
        qkv = self.qkv(x)  # [B, seq, 3*dim]
        qkv[:, :, :self.dim] += delta_q  # Add to Q
        qkv[:, :, -self.dim:] += delta_v  # Add to V

        return qkv

class _LoRA_qkv_timm_eval(nn.Module):
    """
    Evaluation mode LoRA layer with Gumbel-Sparsemax gating.

    Same as training mode, but:
    - No Gumbel noise (deterministic)
    - Lower temperature (sharper selection)
    - No gradient computation
    """
    def __init__(self, task_id, qkv, saved_A, saved_B, t_layer_i, rank, gumbel_gate, save_file):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = rank
        self.task_id = task_id
        self.t_layer_i = t_layer_i

        self.saved_A = saved_A
        self.saved_B = saved_B
        self.gumbel_gate = gumbel_gate
        self.save_file = save_file

    def _compute_normalized_adapter(self, w_a, w_b, x):
        """Compute normalized adapter: (B @ A @ x) / norm (memory-efficient)"""
        adapter_out = w_b(w_a(x))
        norm = torch.norm(w_b.weight) * torch.norm(w_a.weight) + 1e-8
        return adapter_out / norm

    def forward(self, x):
        """Evaluation forward pass (no Gumbel noise)."""
        normalized_adapters_q = []
        normalized_adapters_v = []
        task_indices = []

        # Load all non-pruned tasks
        for i in range(self.task_id):
            # Skip pruned tasks
            if self.gumbel_gate.pruning_mask[i] == 0:
                continue

            task_indices.append(i)

            # Load saved adapters
            saved_A_i = self.saved_A['saved_A_' + str(i)]
            saved_B_i = self.saved_B['saved_B_' + str(i)]

            adapters = list(zip(saved_A_i, saved_B_i))
            A_q, B_q = adapters[self.t_layer_i * 2]
            A_v, B_v = adapters[self.t_layer_i * 2 + 1]

            # Create temporary layers
            w_a_q = nn.Linear(self.dim, self.rank, bias=False)
            w_b_q = nn.Linear(self.rank, self.dim, bias=False)
            w_a_v = nn.Linear(self.dim, self.rank, bias=False)
            w_b_v = nn.Linear(self.rank, self.dim, bias=False)

            w_a_q.weight = nn.Parameter(A_q.weight.to(x.device), requires_grad=False)
            w_b_q.weight = nn.Parameter(B_q.weight.to(x.device), requires_grad=False)
            w_a_v.weight = nn.Parameter(A_v.weight.to(x.device), requires_grad=False)
            w_b_v.weight = nn.Parameter(B_v.weight.to(x.device), requires_grad=False)

            # Normalize
            norm_q = self._compute_normalized_adapter(w_a_q, w_b_q, x)
            norm_v = self._compute_normalized_adapter(w_a_v, w_b_v, x)

            normalized_adapters_q.append(norm_q)
            normalized_adapters_v.append(norm_v)

        # Apply gating (no Gumbel noise, low temp)
        if len(normalized_adapters_q) > 0:
            delta_q, _ = self.gumbel_gate(
                normalized_adapters_q, task_indices, tau=0.5, training=False
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
    


class GumbelGate(nn.Module):
    """
    Gumbel-Sparsemax gating module for sparse task selection.

    Implements Equations 1-2 from Gumbel CL-LoRA:
    - Learnable magnitude parameters α_i for each task
    - Learnable gate logits l_i for Sparsemax selection
    - β = Sparsemax((l + G) / τ) where G ~ Gumbel(0,1)
    - Output: Σ β_i α_i × normalized_adapter_i

    Args:
        max_tasks: Maximum number of tasks (e.g., 20 for CIFAR-100 in 10-task splits)
        init_alpha: Initial value for magnitude parameters α
        init_logit: Initial value for gate logits l
    """
    def __init__(self, max_tasks=20, init_alpha=1.0, init_logit=0.0):
        super().__init__()

        # α: learnable magnitude parameters (one scalar per task)
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(init_alpha)) for _ in range(max_tasks)
        ])

        # l: learnable gate logits (one scalar per task)
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(init_logit)) for _ in range(max_tasks)
        ])

        # Pruning mask: 1 = keep, 0 = pruned (permanent)
        self.register_buffer('pruning_mask', torch.ones(max_tasks))
        self.max_tasks = max_tasks

    def freeze_task_parameters(self, task_id):
        self.alpha[task_id].requires_grad = False
        self.alpha[task_id].grad = None
        self.gate_logits[task_id].requires_grad = False
        self.gate_logits[task_id].grad = None

    def unfreeze_task_parameters(self, task_id):
        self.alpha[task_id].requires_grad = True
        self.gate_logits[task_id].requires_grad = True

    def freeze_alphas(self, task_indices):
        for idx in task_indices:
            self.alpha[idx].requires_grad = False
            self.alpha[idx].grad = None

    def unfreeze_alphas(self, task_indices):
        for idx in task_indices:
            self.alpha[idx].requires_grad = True

    def freeze_logits(self, task_indices):
        for idx in task_indices:
            self.gate_logits[idx].requires_grad = False
            self.gate_logits[idx].grad = None

    def unfreeze_logits(self, task_indices):
        for idx in task_indices:
            self.gate_logits[idx].requires_grad = True

    def forward(self, normalized_adapters, task_indices, tau=1.0, training=True):
        """
        Apply Gumbel-Sparsemax gating.

        Args:
            normalized_adapters: List of normalized adapter outputs [B, seq, dim]
                                 Each is already normalized: adapter / ||A_i B_i||_F
            task_indices: List of task IDs (e.g., [0, 1, 2] for first 3 tasks)
            tau: Temperature for Gumbel-Sparsemax
            training: Whether in training mode (adds Gumbel noise)

        Returns:
            weighted_sum: Σ β_i α_i × normalized_adapter_i, shape [B, seq, dim]
            beta: Sparse gate weights β from Sparsemax, shape [len(task_indices)]
        """
        if len(task_indices) == 0:
            return 0, None

        # Gather logits: [l_0, l_1, ..., l_t]
        logits = torch.stack([self.gate_logits[i] for i in task_indices])  # [N]

        # Apply pruning mask (set pruned tasks to -inf)
        mask = torch.stack([self.pruning_mask[i] for i in task_indices])  # [N]
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # Gumbel-Sparsemax: β = Sparsemax((l + G) / τ)
        beta = gumbel_sparsemax(logits, tau=tau, training=training)  # [N]

        # Gather magnitude parameters: [α_0, α_1, ..., α_t]
        alphas = torch.stack([self.alpha[i] for i in task_indices])  # [N]

        # Weighted sum: Σ β_i α_i × adapter_i
        weighted_sum = 0
        for i, adapter_out in enumerate(normalized_adapters):
            weight = beta[i] * alphas[i]  # β_i * α_i (scalar)
            weighted_sum = weighted_sum + weight * adapter_out

        return weighted_sum, beta

    def get_betas(self, task_indices, tau=1.0):
        """Compute β values without Gumbel noise (for evaluation/decision-making)."""
        if len(task_indices) == 0:
            return torch.tensor([])

        logits = torch.stack([self.gate_logits[i] for i in task_indices])
        mask = torch.stack([self.pruning_mask[i] for i in task_indices])
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # No Gumbel noise (training=False)
        beta = gumbel_sparsemax(logits, tau=tau, training=False)
        return beta

    def get_selection_frequency(self, task_indices, tau=1.0, num_samples=10):
        """
        Sample multiple times from Gumbel-Sparsemax to measure selection robustness.

        Returns:
            mean_betas: Average β values across samples
            selection_freq: Fraction of samples where β > 0 (in Sparsemax support)
        """
        if len(task_indices) == 0:
            return torch.tensor([]), torch.tensor([])

        logits = torch.stack([self.gate_logits[i] for i in task_indices])
        mask = torch.stack([self.pruning_mask[i] for i in task_indices])
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # Sample multiple times with Gumbel noise
        beta_samples = []
        for _ in range(num_samples):
            beta = gumbel_sparsemax(logits, tau=tau, training=True)  # With Gumbel noise
            beta_samples.append(beta)

        beta_samples = torch.stack(beta_samples)  # [num_samples, num_tasks]
        mean_betas = beta_samples.mean(dim=0)
        selection_freq = (beta_samples > 1e-6).float().mean(dim=0)  # Fraction selected

        return mean_betas, selection_freq

    def prune_task(self, task_id):
        """Permanently prune task (set mask to 0)."""
        self.pruning_mask[task_id] = 0

    def keep_task(self, task_id):
        """Keep task (ensure mask is 1)."""
        self.pruning_mask[task_id] = 1


class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param

    def forward(self, x):
        # print('x, param', x.device(), self.pram.device())
        return x * self.param
    
class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.linear_b_q = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_b_q = utils.weight_norm(self.linear_b_q)

    def forward(self, x):
        return self.linear_b_q(x)


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, increment=10, filepath = './', lora_layer=None, eval=False, index=True, cur_task_index=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        self.rank =r
        self.base_vit = copy.deepcopy(vit_model)

        if not eval:
            self.save_file = filepath
            self.increment = increment
            print('save_file', self.save_file)

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        self.w_As, self.w_Bs = [], []  # These are linear layers

        if index:
            print('Initialize task-id and curtask id')
            self.task_id, self.cur_id = 0,0
        
        if cur_task_index != None:
            # print('Update the network!!!', cur_task_index)
            self.task_id = cur_task_index

        # freeze the saved part
        for param in self.base_vit.parameters():
            param.requires_grad = False

        for param in vit_model.parameters():
            param.requires_grad = False

        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        # Init GumbelGate for task selection (replaces scaling factors)
        self.gumbel_gate = GumbelGate(max_tasks=20, init_alpha=1.0, init_logit=0.0)

        # Load pruning mask from previous tasks (enables persistence)
        mask_path = self.save_file + 'pruning_mask.pt'
        if os.path.exists(mask_path):
            self.gumbel_gate.pruning_mask = torch.load(mask_path)
            num_pruned = (self.gumbel_gate.pruning_mask == 0).sum().item()
            print(f'[GumbelGate] Loaded pruning mask: {num_pruned} adapters already pruned')
        else:
            print('[GumbelGate] No previous mask found, starting fresh')

        self.tau = 1.0

        # Do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if not eval:
                blk.attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.gumbel_gate, tau=self.tau
                )
            else:
                blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.gumbel_gate, self.save_file) 

        self.reset_parameters()
        self.lora_vit = vit_model
        if not eval:
            self.lora_vit.head = torch.nn.Identity()
        else:
            self.reset_lora_vit_head()



    def reset_lora_vit_head(self):
        task_incremental = self.increment
        self.lora_vit.head = self.generate_fc(768, (self.task_id)*task_incremental).cuda()
        temp_weights = torch.load(self.save_file+'CLs_weight'+str(self.task_id-1)+'.pt') 
        temp_bias = torch.load(self.save_file+'CLs_bias'+str(self.task_id-1)+'.pt') 

        self.lora_vit.head.weight.data = temp_weights.data.cuda()
        self.lora_vit.head.bias.data = temp_bias.data.cuda()


    # This part is only used during the evaluation
    def reset(self, eval=False):
        self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)

    def reset_parameters(self) -> None:
        # if self.task_id ==0: 
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(w_A.linear_b_q.weight, a=math.sqrt(5) )
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)


    def save_wrap_param(self, filename):
        if self.task_id ==1:   
            scaling_param = torch.zeros(20,20)
        else:
            scaling_param = torch.load(filename + 'scaling_factor'+str(self.task_id-2)+'.pt')
        i = self.task_id-1
        # print('save i', i)
        for j in range(i+1):
            if j == i:
                scaling_param[i][j] = self.wrapped_param[0].param.clone()
            else:
                scaling_param[i][j] = self.wrapped_param_prev[j].param.clone()  
        torch.save(scaling_param, filename + 'scaling_factor'+str(self.task_id-1)+'.pt')
        
    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
           os.makedirs(filename)
        torch.save(self.w_As, filename + 'lora_w_a_'+str(task_id)+'.pt')
        torch.save(self.w_Bs, filename + 'lora_w_b_'+str(task_id)+'.pt')

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def load_eval_vit(self):
        self.lora_vit = copy.deepcopy(self.base_vit)
        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        # for param in self.eval_vit.parameters():
        for param in self.lora_vit.parameters():
            param.requires_grad = False
        
        # for t_layer_i, blk in enumerate(self.eval_vit.blocks):
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank)    
        self.reset_lora_vit_head()

    def compute_ortho_loss(self):
        loss = torch.tensor(0).float().cuda()
        # print('task_id', self.task_id)
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                w_As = torch.load(file_path)
                num_layer = len(self.w_As)
                for j in range(num_layer):
                    temp = torch.matmul(w_As[j].weight.to(self.w_As[j].weight.device), self.w_As[j].weight.t())
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.w_As[j].weight.device)
                    loss += temp
        return loss
    
    def forward(self, x: Tensor, loss= False, eval=False) -> Tensor:
        if eval:
            self.reset(eval=True)
            return self.lora_vit(x)
        elif loss:
            loss = self.compute_ortho_loss()
            return self.lora_vit(x), loss
        else:
            return self.lora_vit(x)
