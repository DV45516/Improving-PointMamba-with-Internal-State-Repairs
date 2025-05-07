import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from tools import builder
from utils.logger import *
from utils import dist_utils
from utils.config import *
from datasets import build_dataset_from_cfg

# Import from your existing code
from tools.mender_finetune import PointMambaWithStates, calculate_internal_correlation

def modified_dataset_builder(args, config):
    """Modified dataset builder to handle different batch size locations"""
    dataset = build_dataset_from_cfg(config._base_, config.others)
    batch_size = getattr(config.others, 'bs', getattr(config, 'total_bs', 8))
    
    shuffle = config.others.subset == 'train'
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=shuffle and not args.distributed,
        num_workers=args.num_workers,
        worker_init_fn=builder.worker_init_fn
    )
    return None, dataloader

def run_diagnosis(args, config, state_key="s_11", samples_per_class=15):
    """Diagnose internal state flaws in PointMamba model"""
    logger = get_logger('diagnosis', os.path.join(args.experiment_path, 'diagnosis_log.txt'))
    os.makedirs(os.path.join(args.experiment_path, 'diagnosis'), exist_ok=True)
    
    # Build dataset and model
    _, test_dataloader = modified_dataset_builder(args, config.dataset.train)
    base_model = builder.model_builder(config.model)
    
    if args.ckpts:
        base_model.load_model_from_ckpt(args.ckpts)
        print_log(f'Loaded model from {args.ckpts}', logger=logger)
    
    # Wrap model with state extraction capability
    wrapped_model = PointMambaWithStates(base_model, state_key=state_key)
    wrapped_model = torch.nn.DataParallel(wrapped_model).cuda()
    
    # Collect correct and incorrect samples
    correct_samples = {}
    incorrect_samples = {}
    wrapped_model.eval()
    
    with torch.no_grad():
        for _, _, data in tqdm(test_dataloader, desc="Finding samples"):
            points = data[0].cuda()
            labels = data[1].cuda()
            
            outputs = wrapped_model(points)
            preds = outputs.argmax(dim=1)
            
            for i, (pred, label) in enumerate(zip(preds, labels)):
                label_item = label.item()
                if pred == label:
                    if label_item not in correct_samples:
                        correct_samples[label_item] = []
                    if len(correct_samples[label_item]) < samples_per_class:
                        correct_samples[label_item].append(points[i:i+1].clone())
                else:
                    if label_item not in incorrect_samples:
                        incorrect_samples[label_item] = []
                    if len(incorrect_samples[label_item]) < samples_per_class:
                        incorrect_samples[label_item].append(points[i:i+1].clone())
    
    # Calculate correlations and ICS scores
    class_diffs = {}
    flaw_metrics = {}
    
    for class_idx in set(correct_samples.keys()) & set(incorrect_samples.keys()):
        if len(correct_samples[class_idx]) < 2 or len(incorrect_samples[class_idx]) < 2:
            continue
            
        # Process correct samples
        correct_corrs = []
        for points in correct_samples[class_idx][:2]:  # Take 2 samples for visualization
            corr = calculate_internal_correlation(wrapped_model, points, class_idx, state_key)
            correct_corrs.append(corr)
            
        # Process incorrect samples
        incorrect_corrs = []
        for points in incorrect_samples[class_idx][:2]:  # Take 2 samples for visualization
            corr = calculate_internal_correlation(wrapped_model, points, class_idx, state_key)
            incorrect_corrs.append(corr)
            
        # Save correlation maps
        if correct_corrs and incorrect_corrs:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot correct samples
            # When plotting the correlations:
            for i, corr in enumerate(correct_corrs):
                corr_np = corr.cpu().numpy()
                # Properly reshape to just show a single row per sample
                axs[0, i].imshow(corr_np.reshape(1, -1), aspect='auto', cmap='viridis')
                axs[0, i].set_title(f"Correct Sample {i+1}")
                axs[0, i].set_xlabel("Feature Dimensions")
                # Remove y-ticks entirely since there's only one sample
                axs[0, i].set_yticks([])
                if i == 0:
                    axs[0, i].set_ylabel("Sample")
            
            # Plot incorrect samples
            for i, corr in enumerate(incorrect_corrs):
                corr_np = corr.cpu().numpy()
                axs[1, i].imshow(corr_np.reshape(1, -1), aspect='auto', cmap='viridis')
                axs[1, i].set_title(f"Incorrect Sample {i+1}")
                axs[1, i].set_xlabel("Feature Dimensions")
                axs[0, i].set_yticks([])
                if i == 0:
                    axs[1, i].set_ylabel("Sample")
            
            plt.suptitle(f"Internal Correlation Patterns for Class {class_idx}")
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(args.experiment_path, f'class_{class_idx}_patterns.png'), dpi=300)
            plt.close(fig)
            
            # Calculate ICS metrics
            correct_corrs_tensor = torch.stack(correct_corrs)
            incorrect_corrs_tensor = torch.stack(incorrect_corrs)
            
            beta = torch.median(torch.cat([correct_corrs_tensor.flatten(), incorrect_corrs_tensor.flatten()]))
            
            # Binarize correlations
            bin_correct = (correct_corrs_tensor > beta).float()
            bin_incorrect = (incorrect_corrs_tensor > beta).float()
            
            # Calculate ICS (simplified for clarity)
            ics_correct = torch.mean(bin_correct).item()
            ics_incorrect = torch.mean(bin_incorrect).item()
            
            # Store the difference
            class_diffs[class_idx] = (ics_correct - ics_incorrect)
            
            # Store metrics for JSON
            flaw_metrics[str(class_idx)] = {
                "ics_correct": ics_correct,
                "ics_incorrect": ics_incorrect,
                "ics_diff": (ics_correct - ics_incorrect),
                "beta_threshold": beta.item()
            }
    
    # Save flaw metrics to JSON
    with open(os.path.join(args.experiment_path, 'flaw_metrics.json'), 'w') as f:
        json.dump(flaw_metrics, f, indent=4)
    
    # Create ICS comparison visualization
    plt.figure(figsize=(12, 6))
    classes = list(class_diffs.keys())
    diffs = [class_diffs[c] for c in classes]
    
    plt.bar(range(len(classes)), diffs)
    plt.xticks(range(len(classes)), [str(c) for c in classes])
    plt.xlabel('Class')
    plt.ylabel('ICS Difference (Correct - Incorrect)')
    plt.title('ICS Difference by Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.experiment_path, 'ics_differences.png'), dpi=300)
    plt.close()
    
    # Save ICS comparison components
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    correct_ics = [flaw_metrics[str(c)]["ics_correct"] for c in classes]
    incorrect_ics = [flaw_metrics[str(c)]["ics_incorrect"] for c in classes]
    
    plt.bar(x - width/2, correct_ics, width, label='Correct Samples')
    plt.bar(x + width/2, incorrect_ics, width, label='Incorrect Samples')
    
    plt.xlabel('Class')
    plt.ylabel('ICS Score')
    plt.title('ICS Components by Class')
    plt.xticks(x, [str(c) for c in classes])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.experiment_path, 'ics_components.png'), dpi=300)
    plt.close()
    
    # Save ICS comparison visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(classes)), diffs)
    plt.axhline(y=0.2, color='r', linestyle='--', label='Threshold (0.2)')
    plt.xticks(range(len(classes)), [str(c) for c in classes])
    plt.xlabel('Class')
    plt.ylabel('ICS Difference (Correct - Incorrect)')
    plt.title('ICS Difference by Class')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.experiment_path, 'ics_comparison.png'), dpi=300)
    plt.close()
    
    # Save ICS differences as numpy file
    np.save(os.path.join(args.experiment_path, 'ics_diff.npy'), np.array(diffs))
    
    # Make diagnosis
    avg_diff = sum(class_diffs.values()) / len(class_diffs) if class_diffs else 0
    print_log(f"Average ICS difference: {avg_diff:.4f}", logger=logger)
    
    if avg_diff > 0.05:
        print_log("DIAGNOSIS: Significant internal state flaws detected!", logger=logger)
        print_log("Recommendation: Internal state repair is recommended", logger=logger)
    else:
        print_log("DIAGNOSIS: No significant internal state flaws detected", logger=logger)
        print_log("Recommendation: Internal state repair is NOT recommended", logger=logger)
    
    return class_diffs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('PointMamba Internal State Flaw Diagnosis')
    parser.add_argument('--config', type=str, default='cfgs/finetune_modelnet.yaml', help='Path to config file')
    parser.add_argument('--ckpts', type=str, default='pre_trained_models/finetune_modelnet.pth', help='Path to checkpoint')
    parser.add_argument('--state_key', type=str, default='s_11', help='State key to analyze (e.g., s_11, x_5)')
    parser.add_argument('--samples_per_class', type=int, default=20, help='Number of samples per class to analyze')
    parser.add_argument('--experiment_path', type=str, default='./diagnosis_experiment', help='Path to save results')
    parser.add_argument('--log_name', type=str, default='diagnosis', help='Logger name')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--launcher', type=str, default='pytorch', help='Launcher type')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--distributed', action='store_true', default=False, help='Use distributed training')
    
    args = parser.parse_args()
    
    # Initialize distributed training if needed
    if args.distributed:
        dist_utils.init_dist(args.launcher)
        torch.cuda.set_device(args.local_rank)
    
    # Create experiment path if it doesn't exist
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
    
    # Set up logger
    logger = get_logger(args.log_name, os.path.join(args.experiment_path, 'diagnosis_log.txt'))
    
    # Load config
    config = cfg_from_yaml_file(args.config)
    
    # Print configuration
    print_log(f"=== Configuration ===", logger=logger)
    print_log(f"Checkpoint: {args.ckpts}", logger=logger)
    print_log(f"State key: {args.state_key}", logger=logger)
    print_log(f"Samples per class: {args.samples_per_class}", logger=logger)
    
    results = run_diagnosis(args, config, args.state_key)