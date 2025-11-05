import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.gumbel_utils import TemperatureScheduler

import timm
from backbone.lora import LoRA_ViT_timm
import torch.distributed as dist

import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[self._device], output_device=self._device, find_unused_parameters=True)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        # to test
        # self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def update_network(self, index=True):
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)

        # if use DINO
        # model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)

        # SD-LoRA-RR
        '''
        if self._cur_task >=4 and self._cur_task <8:
            rank = 8 #8
        elif self._cur_task >=8:
            rank = 6 #6
        # elif self._cur_task >=8:
        #     rank = 4
        else:
            rank = 10
        '''
        rank=10
        model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'], 
        cur_task_index= self._cur_task)
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                # weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args['filepath']

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                # lr=self.init_lr, 
                self.args["lrate"],
                # weight_decay=self.weight_decay
                betas=(0.9, 0.999)
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler



    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        # Initialize temperature scheduler for Gumbel-Sparsemax
        tau_init = self.args.get("gumbel_tau_init", 5.0)
        tau_final = self.args.get("gumbel_tau_final", 0.5)
        anneal_rate = self.args.get("gumbel_anneal_rate", 0.999)
        lambda_sparsity = self.args.get("lambda_sparsity", 0.01)

        temp_scheduler = TemperatureScheduler(
            tau_init=tau_init, tau_final=tau_final, anneal_rate=anneal_rate
        )

        logging.info(f"[Gumbel CL-LoRA] tau_init={tau_init}, tau_final={tau_final}, "
                    f"anneal_rate={anneal_rate}, lambda_sparsity={lambda_sparsity}")

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_clf = 0.0
            losses_sparsity = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                # Get current temperature
                tau = temp_scheduler.step()

                # Update tau in backbone (needed for forward pass)
                if len(self._multiple_gpus) > 1:
                    self._network.module.backbone.tau = tau
                else:
                    self._network.backbone.tau = tau

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits['logits']

                # Classification loss
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )

                # Sparsity regularization loss (Equation 3)
                # Collect sparsity losses from all LoRA layers
                sparsity_loss_total = 0
                if len(self._multiple_gpus) > 1:
                    backbone = self._network.module.backbone
                else:
                    backbone = self._network.backbone

                # Iterate through LoRA layers to collect sparsity loss
                for blk in backbone.lora_vit.blocks:
                    if hasattr(blk.attn.qkv, 'get_sparsity_loss'):
                        sparsity_loss_total += blk.attn.qkv.get_sparsity_loss()

                # Total loss (Equation 3 from paper)
                # loss = loss_clf + lambda_sparsity * sparsity_loss + ortho_weight * ortho_loss
                loss = loss_clf + lambda_sparsity * sparsity_loss_total

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_sparsity += sparsity_loss_total.item() if isinstance(sparsity_loss_total, torch.Tensor) else 0

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f} (clf={:.3f}, sparse={:.4f}), tau={:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_sparsity / len(train_loader),
                    tau,
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f} (clf={:.3f}, sparse={:.4f}), tau={:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_sparsity / len(train_loader),
                    tau,
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

        # Conditional growth decision (Equation 4)
        self._conditional_growth_decision()

    def _conditional_growth_decision(self):
        """
        Conditional growth mechanism: Trust sparsemax to identify useful adapters.

        Sparsemax learns which adapters are useful and sets others to zero.
        We make those zeros permanent to save memory (sublinear growth).
        """
        # Trust sparsemax: only prune true zeros (plus small epsilon for numerical stability)
        threshold = self.args.get("growth_threshold", 1e-6)
        tau_eval = self.args.get("gumbel_tau_final", 0.5)

        # Get backbone
        if len(self._multiple_gpus) > 1:
            backbone = self._network.module.backbone
        else:
            backbone = self._network.backbone

        gumbel_gate = backbone.gumbel_gate

        # Evaluate all task betas
        task_indices = list(range(self._cur_task + 1))
        beta_values = gumbel_gate.get_betas(task_indices, tau=tau_eval)

        logging.info(f"\n[Conditional Growth] Task {self._cur_task} - Beta values: {beta_values.detach().cpu().numpy()}")
        logging.info(f"[Conditional Growth] Pruning threshold: {threshold:.1e} (sparsemax zeros)")

        # Prune adapters where sparsemax → 0
        num_pruned = 0
        for i, task_idx in enumerate(task_indices):
            beta_val = beta_values[i].item()

            # Skip already pruned
            if gumbel_gate.pruning_mask[task_idx] == 0:
                continue

            if beta_val < threshold:
                # Sparsemax set to zero → prune permanently
                gumbel_gate.prune_task(task_idx)
                num_pruned += 1
                logging.info(f"[Conditional Growth] ✗ Pruned task {task_idx} (beta={beta_val:.6f} ≈ 0)")

        # Statistics
        num_active = (gumbel_gate.pruning_mask[:self._cur_task+1] > 0).sum().item()
        num_total = self._cur_task + 1
        logging.info(f"[Conditional Growth] Active: {num_active}/{num_total} ({100*num_active/num_total:.1f}%)")

        if num_active < num_total:
            logging.info(f"[Conditional Growth] 🎯 Sublinear growth: {num_total - num_active} adapters pruned")

        # Save pruning mask for next task (CRITICAL: persistence across tasks)
        mask_path = self.args['filepath'] + 'pruning_mask.pt'
        torch.save(gumbel_gate.pruning_mask.cpu(), mask_path)
        logging.info(f"[Conditional Growth] Pruning mask saved to {mask_path}")

        # Save beta values for analysis
        beta_path = self.args['filepath'] + f'beta_values_task_{self._cur_task}.pt'
        torch.save({
            'beta_values': beta_values.detach().cpu(),
            'task_indices': task_indices,
            'pruning_mask': gumbel_gate.pruning_mask.cpu(),
            'threshold': threshold,
        }, beta_path)
