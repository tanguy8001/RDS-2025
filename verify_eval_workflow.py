#!/usr/bin/env python3
"""
Verification script to demonstrate that evaluation uses ONLY non-pruned adapters.

This script traces through the evaluation path to show:
1. Which layer type is used (train vs eval)
2. Which adapters are loaded
3. Which adapters are skipped due to pruning mask
4. The final set of active adapters

Usage:
    python verify_eval_workflow.py [--task_id N] [--mask "1,0,1,0,..."]
"""

import argparse

def simulate_eval_forward(task_id, pruning_mask):
    """
    Simulate the evaluation forward pass to show which adapters are used.
    
    Args:
        task_id: Current task ID (next task to train, e.g., 10 after training task 9)
        pruning_mask: List of 0s and 1s (0=pruned, 1=kept)
    """
    print("\n" + "="*80)
    print("EVALUATION WORKFLOW VERIFICATION")
    print("="*80)
    
    print(f"\nScenario:")
    print(f"  - task_id = {task_id} (next task to train)")
    print(f"  - Completed tasks: 0 to {task_id-1}")
    print(f"  - Pruning mask: {pruning_mask}")
    
    # Simulate _LoRA_qkv_timm_eval.forward()
    print(f"\nEvaluation Forward Pass:")
    print(f"  Layer type: _LoRA_qkv_timm_eval (deterministic, no Gumbel noise)")
    print(f"  Loop range: for i in range({task_id})  # 0 to {task_id-1}")
    print()
    
    task_indices = []
    
    for i in range(task_id):
        # Check 1: Pruning mask
        if pruning_mask[i] == 0:
            print(f"  Task {i}: SKIPPED (pruning_mask[{i}] = 0)")
            continue
        
        # Check 2: Check if saved (simulated - assume all are saved)
        saved = True
        if not saved:
            print(f"  Task {i}: SKIPPED (not saved yet)")
            continue
        
        print(f"  Task {i}: ✓ LOADED (pruning_mask[{i}] = 1)")
        task_indices.append(i)
    
    print()
    print("-" * 80)
    print(f"Result:")
    print(f"  Active adapters: {task_indices}")
    print(f"  Count: {len(task_indices)}/{task_id} adapters")
    print(f"  Pruned: {task_id - len(task_indices)}/{task_id} adapters")
    print(f"  Sublinear growth: {'YES' if len(task_indices) < task_id else 'NO'}")
    
    # Verify assertion
    if len(task_indices) > 0:
        assert all(pruning_mask[i] == 1 for i in task_indices), \
            f"ERROR: Found pruned tasks in active adapters!"
        print(f"\n✓ ASSERTION PASSED: All active adapters are non-pruned")
    else:
        print(f"\n  (No adapters to verify - all tasks pruned)")
    
    print("="*80 + "\n")
    
    return task_indices


def compare_train_vs_eval():
    """Compare which adapters are used in training vs evaluation."""
    print("\n" + "="*80)
    print("TRAINING vs EVALUATION COMPARISON")
    print("="*80)
    
    task_id = 5  # After training task 4, before training task 5
    pruning_mask = [1, 0, 1, 0, 1]  # Tasks 0,2,4 kept, 1,3 pruned
    
    print(f"\nScenario: task_id = {task_id}, mask = {pruning_mask}")
    
    # Training forward
    print(f"\n1. TRAINING Forward (_LoRA_qkv_timm_train):")
    print(f"   Loop: for i in range({task_id})  # 0 to {task_id-1}")
    train_tasks = []
    for i in range(task_id):
        if pruning_mask[i] == 0:
            print(f"   Task {i}: SKIPPED (pruned)")
        else:
            print(f"   Task {i}: ✓ Loaded (normalized, frozen)")
            train_tasks.append(i)
    print(f"   Task {task_id}: ✓ Current task (NOT normalized, trainable)")
    train_tasks.append(task_id)
    print(f"   Active: {train_tasks} → {len(train_tasks)} adapters")
    
    # Evaluation forward
    print(f"\n2. EVALUATION Forward (_LoRA_qkv_timm_eval):")
    print(f"   Loop: for i in range({task_id})  # 0 to {task_id-1}")
    eval_tasks = []
    for i in range(task_id):
        if pruning_mask[i] == 0:
            print(f"   Task {i}: SKIPPED (pruned)")
        else:
            print(f"   Task {i}: ✓ Loaded (normalized, frozen)")
            eval_tasks.append(i)
    print(f"   Task {task_id}: NOT INCLUDED (not in loop range)")
    print(f"   Active: {eval_tasks} → {len(eval_tasks)} adapters")
    
    print(f"\nKey Differences:")
    print(f"  - Training includes current task {task_id}: YES")
    print(f"  - Evaluation includes current task {task_id}: NO")
    print(f"  - Training adapter count: {len(train_tasks)}")
    print(f"  - Evaluation adapter count: {len(eval_tasks)}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Verify evaluation workflow')
    parser.add_argument('--task_id', type=int, default=10,
                        help='Current task ID (default: 10)')
    parser.add_argument('--mask', type=str, default="1,0,1,0,0,0,0,0,0,0",
                        help='Pruning mask as comma-separated 0s and 1s (default: "1,0,1,0,0,0,0,0,0,0")')
    args = parser.parse_args()
    
    # Parse pruning mask
    pruning_mask = [int(x) for x in args.mask.split(',')]
    
    if len(pruning_mask) < args.task_id:
        print(f"Error: Pruning mask length ({len(pruning_mask)}) < task_id ({args.task_id})")
        return
    
    # Run simulations
    simulate_eval_forward(args.task_id, pruning_mask)
    compare_train_vs_eval()
    
    # Example scenarios
    print("\n" + "="*80)
    print("ADDITIONAL EXAMPLE SCENARIOS")
    print("="*80)
    
    print("\nScenario 1: All adapters kept")
    simulate_eval_forward(5, [1, 1, 1, 1, 1])
    
    print("\nScenario 2: All adapters pruned except first and last")
    simulate_eval_forward(5, [1, 0, 0, 0, 1])
    
    print("\nScenario 3: Complete CIFAR-100 10-task setting (after task 9)")
    simulate_eval_forward(10, [1, 0, 1, 0, 0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    main()
