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
from utils.gumbel_utils import gumbel_sparsemax, sparsity_loss

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
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v,
        task_id, saved_A, saved_B, t_layer_i, rank, gumbel_gate, tau=1.0, eval1=False):
        super().__init__()
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()

        # Replace scaling factors with GumbelGate
        self.gumbel_gate = gumbel_gate
        self.tau = tau

        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.eval = eval1

        # Store sparsity loss for backprop
        self.sparsity_loss_q = 0
        self.sparsity_loss_v = 0

    def forward(self, x, tau=None):
        """
        Forward pass with Gumbel-Sparsemax gating.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            tau: Temperature (if None, uses self.tau)

        Returns:
            qkv: Output tensor [batch_size, seq_len, 3*dim]
        """
        if tau is None:
            tau = self.tau

        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)

        # Collect normalized adapter outputs for previous tasks
        adapter_outputs_q = []
        adapter_outputs_v = []
        task_indices = list(range(self.task_id))

        for i in range(self.task_id):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i, saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            # Load Q adapter
            w_a_linear_q.weight = Parameter(A_q.weight)
            w_a_linear_q.weight.requires_grad = False
            w_a_linear_q.to(x.device)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_b_linear_q.weight.requires_grad = False
            w_b_linear_q.to(x.device)

            # Load V adapter
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_a_linear_v.weight.requires_grad = False
            w_a_linear_v.to(x.device)
            w_b_linear_v.weight = Parameter(B_v.weight)
            w_b_linear_v.weight.requires_grad = False
            w_b_linear_v.to(x.device)

            # Compute normalized adapter output (Equation 1: direction decoupling)
            norm_q = torch.norm(w_b_linear_q.weight) * torch.norm(w_a_linear_q.weight)
            norm_v = torch.norm(w_b_linear_v.weight) * torch.norm(w_a_linear_v.weight)

            adapter_out_q = w_b_linear_q(w_a_linear_q(x)) / norm_q
            adapter_out_v = w_b_linear_v(w_a_linear_v(x)) / norm_v

            adapter_outputs_q.append(adapter_out_q)
            adapter_outputs_v.append(adapter_out_v)

        # Apply Gumbel-Sparsemax gating to previous tasks (Equation 2)
        if len(adapter_outputs_q) > 0:
            new_q, beta_q, sparsity_loss_q = self.gumbel_gate(
                adapter_outputs_q, task_indices, tau=tau, hard=False, training=self.training
            )
            new_v, beta_v, sparsity_loss_v = self.gumbel_gate(
                adapter_outputs_v, task_indices, tau=tau, hard=False, training=self.training
            )
            # Store sparsity losses for regularization
            self.sparsity_loss_q = sparsity_loss_q
            self.sparsity_loss_v = sparsity_loss_v
        else:
            new_q, new_v = 0, 0
            self.sparsity_loss_q = 0
            self.sparsity_loss_v = 0

        # Add current task adapter (with its own alpha from gumbel_gate)
        # For current task, we use the alpha parameter directly
        current_alpha_q = self.gumbel_gate.alpha[self.task_id]
        current_alpha_v = self.gumbel_gate.alpha[self.task_id]

        new_q = new_q + current_alpha_q * self.linear_b_q(self.linear_a_q(x))
        new_v = new_v + current_alpha_v * self.linear_b_v(self.linear_a_v(x))

        # Apply to QKV
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

    def get_sparsity_loss(self):
        """Return total sparsity regularization loss."""
        return self.sparsity_loss_q + self.sparsity_loss_v

class _LoRA_qkv_timm_eval(nn.Module):
    """
    Evaluation mode LoRA layer with Gumbel-Sparsemax gating.

    In timm it is implemented as:
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(self, task_id, qkv: nn.Module, saved_A, saved_B, t_layer_i, rank, gumbel_gate, save_file):
        super().__init__()
        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.save_file = save_file
        self.gumbel_gate = gumbel_gate

    def forward(self, x):
        """
        Evaluation forward pass with Gumbel gating (no Gumbel noise).
        """
        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)

        # Collect normalized adapter outputs
        adapter_outputs_q = []
        adapter_outputs_v = []
        task_indices = list(range(self.task_id))

        for i in range(self.task_id):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i, saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_b_linear_v.weight = Parameter(B_v.weight)

            # Compute normalized adapter output
            norm_q = torch.norm(w_b_linear_q.weight) * torch.norm(w_a_linear_q.weight)
            norm_v = torch.norm(w_b_linear_v.weight) * torch.norm(w_a_linear_v.weight)

            adapter_out_q = w_b_linear_q(w_a_linear_q(x)) / norm_q
            adapter_out_v = w_b_linear_v(w_a_linear_v(x)) / norm_v

            adapter_outputs_q.append(adapter_out_q)
            adapter_outputs_v.append(adapter_out_v)

        # Apply Gumbel-Sparsemax gating (no noise in eval mode)
        if len(adapter_outputs_q) > 0:
            new_q, _, _ = self.gumbel_gate(
                adapter_outputs_q, task_indices, tau=0.5, hard=False, training=False
            )
            new_v, _, _ = self.gumbel_gate(
                adapter_outputs_v, task_indices, tau=0.5, hard=False, training=False
            )
        else:
            new_q, new_v = 0, 0

        # Note: Current task adapter is not included in eval mode
        # (only previous tasks are aggregated)

        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    


class GumbelGate(nn.Module):
    """
    Gumbel-Sparsemax gating module for CL-LoRA task selection.

    Maintains learnable gate logits and magnitude parameters (alpha, beta)
    for each task, implementing Equations 1-2 from the Gumbel CL-LoRA paper.

    Args:
        max_tasks: Maximum number of tasks (e.g., 20 for CIFAR-100)
        init_alpha: Initial value for magnitude parameters
        init_logit: Initial value for gate logits
    """
    def __init__(self, max_tasks=20, init_alpha=0.8, init_logit=0.0):
        super(GumbelGate, self).__init__()

        # Alpha: learnable magnitude parameters (one per task)
        # These replace the fixed scaling_factor in original SD-LoRA
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor([init_alpha])) for _ in range(max_tasks)
        ])

        # Gate logits: learnable affinity scores (one per task)
        # These determine task selection via Gumbel-Sparsemax
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.tensor([init_logit])) for _ in range(max_tasks)
        ])

        # Pruning mask: binary mask for conditional growth (Equation 4)
        # Once set to 0, the adapter is permanently pruned
        self.register_buffer('pruning_mask', torch.ones(max_tasks))

        self.max_tasks = max_tasks

    def forward(self, x, task_indices, tau=1.0, hard=False, training=True):
        """
        Apply Gumbel-Sparsemax gating to select and scale adapters.

        Args:
            x: List of adapter outputs, one per task
               Each element has shape [batch_size, seq_len, dim]
            task_indices: List of task IDs to consider (e.g., [0, 1, 2] for first 3 tasks)
            tau: Temperature for Gumbel-Sparsemax
            hard: If True, use hard (one-hot) selection
            training: Whether in training mode

        Returns:
            weighted_output: Weighted sum of adapter outputs, shape [batch_size, seq_len, dim]
            beta: Task selection weights, shape [num_active_tasks]
            sparsity_reg: Sparsity regularization loss (scalar)
        """
        num_active_tasks = len(task_indices)

        if num_active_tasks == 0:
            # No previous tasks
            return 0, None, 0

        # Gather logits for active tasks
        logits = torch.cat([self.gate_logits[i] for i in task_indices], dim=0)  # [num_active_tasks]

        # Apply pruning mask (set pruned tasks to -inf)
        mask = torch.cat([self.pruning_mask[i:i+1] for i in task_indices], dim=0)  # [num_active_tasks]
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # Gumbel-Sparsemax selection (Equation 2)
        beta = gumbel_sparsemax(logits, tau=tau, hard=hard, training=training)  # [num_active_tasks]

        # Gather alpha (magnitude) parameters
        alphas = torch.cat([self.alpha[i] for i in task_indices], dim=0)  # [num_active_tasks]

        # Compute weighted sum: Σ β_i * α_i * LoRA_i(x)
        # x is a list of tensors, each [batch, seq, dim]
        weighted_output = 0
        for i, adapter_output in enumerate(x):
            # beta[i] is scalar, alphas[i] is scalar
            weight = beta[i] * alphas[i]
            weighted_output = weighted_output + weight * adapter_output

        # Sparsity regularization (Equation 3)
        sparsity_reg = sparsity_loss(beta) if training else torch.tensor(0.0, device=beta.device)

        return weighted_output, beta, sparsity_reg

    def get_betas(self, task_indices, tau=1.0):
        """
        Get current beta values (for logging/pruning decisions).

        Args:
            task_indices: List of task IDs
            tau: Temperature

        Returns:
            beta: Task selection weights
        """
        if len(task_indices) == 0:
            return torch.tensor([])

        logits = torch.cat([self.gate_logits[i] for i in task_indices], dim=0)
        mask = torch.cat([self.pruning_mask[i:i+1] for i in task_indices], dim=0)
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # No Gumbel noise, just sparsemax
        beta = gumbel_sparsemax(logits, tau=tau, hard=False, training=False)
        return beta

    def prune_task(self, task_id):
        """
        Permanently prune a task by setting its mask to 0.

        Args:
            task_id: Task ID to prune
        """
        self.pruning_mask[task_id] = 0
        print(f"[GumbelGate] Pruned task {task_id} (beta below threshold)")

    def keep_task(self, task_id):
        """
        Keep a task (set mask to 1).

        Args:
            task_id: Task ID to keep
        """
        self.pruning_mask[task_id] = 1
        print(f"[GumbelGate] Keeping task {task_id} (beta above threshold)")


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

        # Initialize GumbelGate for task selection (replaces scaling factors)
        self.gumbel_gate = GumbelGate(max_tasks=20, init_alpha=0.8, init_logit=0.0)

        # Load pruning mask from previous tasks (enables persistence)
        mask_path = self.save_file + 'pruning_mask.pt'
        if os.path.exists(mask_path):
            self.gumbel_gate.pruning_mask = torch.load(mask_path)
            num_pruned = (self.gumbel_gate.pruning_mask == 0).sum().item()
            print(f'[GumbelGate] Loaded pruning mask: {num_pruned} adapters already pruned')
        else:
            print('[GumbelGate] No previous mask found, starting fresh')

        # Temperature for Gumbel-Sparsemax (will be updated by learner)
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
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.gumbel_gate, tau=self.tau, eval1=False
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
