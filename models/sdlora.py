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
from utils.gumbel_utils import TemperatureScheduler, sparsity_loss

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

            #optimizer = optim.SGD(
            #    self._network.parameters(),
            #    lr=self.args["lrate"],
            #    momentum=0.9,
            #)  # 1e-5
            optimizer = optim.AdamW(
                self._network.parameters(),
                lr=self.args["beta_lr"],
                weight_decay=self.args["weight_decay"]
            )
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
        backbone = self._network.module.backbone if len(self._multiple_gpus) > 1 else self._network.backbone
        gumbel_gate = backbone.gumbel_gate
        task_indices = list(range(self._cur_task + 1))

        temp_scheduler = TemperatureScheduler(self.args["gumbel_tau_init"], self.args["gumbel_tau_final"], self.args["gumbel_anneal_rate"])

        total_epochs = self.args["epochs"]
        magnitude_epochs = int(self.args["alpha_beta_ratio"] * total_epochs)
        selection_epochs = total_epochs - magnitude_epochs

        logging.info(f"[Two-Phase Training] Task {self._cur_task}: Selection = {selection_epochs} epochs, "
                    f"Magnitude = {magnitude_epochs} epochs")

        logging.info(f"[Phase Magnitude] Learning magnitude (new adapter trainable, α trainable, l frozen)")
        
        # Unfreeze new task's AB adapters
        for param in backbone.w_As:
            param.weight.requires_grad = True
        for param in backbone.w_Bs:
            param.weight.requires_grad = True

        gumbel_gate.freeze_logits(task_indices)
        gumbel_gate.unfreeze_alphas(task_indices)

        # New optimizer with param groups
        alpha_params = [gumbel_gate.alpha[i] for i in task_indices]
        lora_params = [p.weight for p in backbone.w_As + backbone.w_Bs]

        optimizer_p2 = optim.SGD([
            {'params': lora_params, 'lr': self.args["lrate"]},
            {'params': alpha_params, 'lr': self.args["alpha_lr"]}
        ], momentum=0.9)

        scheduler_p2 = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer_p2,
            milestones=[int(0.6 * magnitude_epochs), int(0.8 * magnitude_epochs)],
            gamma=self.args["lrate_decay"] if self.args["lrate_decay"] > 0 else 0.5
        )

        self._train_phase(train_loader, test_loader, optimizer_p2, scheduler_p2, temp_scheduler,
                         self.args["lambda_sparsity"], phase=2, epochs=magnitude_epochs)

        logging.info(f"[Phase Selection] Learning selection (new adapter frozen, α frozen, l trainable)")

        # Freeze new task's AB adapters
        for param in backbone.w_As:
            param.weight.requires_grad = False
            param.weight.grad = None
        for param in backbone.w_Bs:
            param.weight.requires_grad = False
            param.weight.grad = None

        gumbel_gate.freeze_alphas(task_indices)
        gumbel_gate.unfreeze_logits(task_indices)

        self._train_phase(train_loader, test_loader, optimizer, scheduler, temp_scheduler,
                         self.args["lambda_sparsity"], phase=1, epochs=selection_epochs)

        logging.info(f'[After Both Phases] LoRA LR: {self.args["lrate"]:.2e}, Alpha LR: {self.args.get("alpha_lr", self.args["lrate"]):.2e}')

        self._conditional_growth_decision()

    def _train_phase(self, train_loader, test_loader, optimizer, scheduler, temp_scheduler,
                     lambda_sparsity, phase, epochs):
        """Train for one phase (either selection or magnitude learning)."""
        backbone = self._network.module.backbone if len(self._multiple_gpus) > 1 else self._network.backbone
        blocks = backbone.lora_vit.blocks
        gumbel_gate = backbone.gumbel_gate

        prog_bar = tqdm(range(epochs), desc=f"Phase {phase}")
        for epoch in prog_bar:
            self._network.train()
            losses, correct, total = 0.0, 0, 0

            for _, inputs, targets in train_loader:
                tau = temp_scheduler.step()
                backbone.tau = tau

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, ortho_loss=True)[0]['logits']

                # Classification loss
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)

                # Reg losses
                loss_reg = 0

                # Phase 1: Sparsity reg (encourage sparse β selection)
                if phase == 1:
                    num_layers = 0
                    for blk in blocks:
                        qkv_layer = blk.attn.qkv
                        if hasattr(qkv_layer, 'last_beta_q') and qkv_layer.last_beta_q is not None:
                            loss_reg += sparsity_loss(qkv_layer.last_beta_q)
                            loss_reg += sparsity_loss(qkv_layer.last_beta_v)
                            num_layers += 1
                    if num_layers > 0:
                        loss_reg = lambda_sparsity * (loss_reg / (2 * num_layers))

                # Phase 2: No alpha regularization (let them learn freely)
                # Use gradient clipping instead to prevent explosion

                loss = loss_clf + loss_reg

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for alphas (prevents explosion without constraining values)
                if phase == 2:
                    alpha_params = [gumbel_gate.alpha[i] for i in range(self._cur_task + 1)]
                    torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=1.0)

                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = f"Phase {phase}, Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, " \
                       f"tau={tau:.3f}, Train {train_acc:.2f}%, Test {test_acc:.2f}%"
            else:
                info = f"Phase {phase}, Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, " \
                       f"tau={tau:.3f}, Train {train_acc:.2f}%"

            prog_bar.set_description(info)

        logging.info(info)

    def _conditional_growth_decision(self):
        backbone = self._network.module.backbone if len(self._multiple_gpus) > 1 else self._network.backbone
        gumbel_gate = backbone.gumbel_gate
        current_task = self._cur_task
        task_indices = list(range(current_task + 1))

        with torch.no_grad():
            betas = gumbel_gate.get_betas(task_indices, tau=0.3)

        current_beta = betas[-1].item()

        logging.info(f"\n{'='*60}")
        logging.info(f"[Conditional Growth] Task {current_task} | Current Beta: {current_beta:.3f}")
        logging.info(f"{'='*60}")
        logging.info(f"{'Task':<6} {'α':<8} {'logit':<8} {'β':<10} {'Decision':<10}")
        logging.info("-" * 60)

        for idx, task_idx in enumerate(task_indices):
            alpha = gumbel_gate.alpha[task_idx].item()
            logit = gumbel_gate.gate_logits[task_idx].item()
            beta = betas[idx].item()

            if task_idx == current_task:
                decision = "✗ PRUNE" if current_beta < 0.05 else "✓ KEEP"
                logging.info(f"{task_idx:<6} {alpha:<8.3f} {logit:<8.3f} {beta:<10.4f} {decision:<10}")
            else:
                mask = int(gumbel_gate.pruning_mask[task_idx].item())
                logging.info(f"{task_idx:<6} {alpha:<8.3f} {logit:<8.3f} {beta:<10.4f} mask={mask}")

        # Make decision
        if current_beta < 0.05:
            gumbel_gate.prune_task(current_task)
            logging.info(f"\n✗ PRUNED: beta={current_beta:.2f} < 0.05")
        else:
            gumbel_gate.keep_task(current_task)
            logging.info(f"\n✓ KEPT: beta={current_beta:.2f} >= 0.05")

        gumbel_gate.freeze_task_parameters(current_task)

        # Summary
        num_active = (gumbel_gate.pruning_mask[:current_task + 1] > 0).sum().item()
        logging.info(f"Active: {num_active}/{current_task + 1} adapters")

        mask_path = self.args['filepath'] + 'pruning_mask.pt'
        torch.save(gumbel_gate.pruning_mask.cpu(), mask_path)

        state_path = self.args['filepath'] + f'gumbel_gate_task_{current_task}.pt'
        torch.save({
            'alphas': [gumbel_gate.alpha[i].detach().cpu() for i in range(current_task + 1)],
            'gate_logits': [gumbel_gate.gate_logits[i].detach().cpu() for i in range(current_task + 1)],
            'pruning_mask': gumbel_gate.pruning_mask.cpu(),
            'task_id': current_task,
        }, state_path)
        logging.info(f"{'='*60}\n")
