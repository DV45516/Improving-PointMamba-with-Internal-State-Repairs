import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
import time

from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
    ]
)

train_transforms_raw = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


class PointMambaWithStates(nn.Module):
    """Wrapper for PointMamba that extracts internal states"""
    def __init__(self, model, state_key="s_11"):
        super().__init__()
        self.model = model
        self.state_hooks = []
        self.stored_states = {}
        self.state_key = state_key
        
        # Parse the state key
        parts = self.state_key.split('_')
        if len(parts) == 2:
            self.state_type = parts[0]  # 'x' or 's'
            self.target_layer = int(parts[1])
            print(f"Targeting {self.state_type} state in layer {self.target_layer}")
        else:
            print(f"Invalid state key format: {state_key}, should be like 's_11' or 's_11'")
            self.state_type = 's'  # Default to 's'
            self.target_layer = 11  # Default to layer 11
        
        # Save original forward method
        self.original_mixer_forward = None
        # Install the hooks
        self.install_hooks()
    
    def install_hooks(self):
        """Install hooks by modifying the forward method of the target mixer"""
        if not hasattr(self.model, 'blocks') or not hasattr(self.model.blocks, 'layers'):
            print("ERROR: Model structure doesn't match expected pattern")
            return
        
        # Only modify the target layer
        blocks = self.model.blocks.layers
        if self.target_layer >= len(blocks):
            print(f"ERROR: Target layer {self.target_layer} is out of range")
            return
        
        target_block = blocks[self.target_layer]
        if not hasattr(target_block, 'mixer'):
            print(f"ERROR: Block {self.target_layer} doesn't have the expected structure")
            return
        
        # Store original method
        mixer = target_block.mixer
        self.original_mixer_forward = mixer.forward
        
        # Create a new forward method that captures the state
        def patched_forward(self_mixer, x, **kwargs):
            # Check which state type we're looking for
            state_type = self.state_key.split('_')[0]
            
            if state_type == 'x':
                # Capture the input projection (in_proj output)
                proj = self_mixer.in_proj(x)
                state_name = f"x_{self.target_layer}"
                if state_name not in self.stored_states:
                    self.stored_states[state_name] = []
                self.stored_states[state_name].append(proj)
                
            # Call the original forward to get the result
            result = self.original_mixer_forward(x, **kwargs)
            
            if state_type == 's':
                # Capture the SSM output (the 's' state)
                state_name = f"s_{self.target_layer}"
                if state_name not in self.stored_states:
                    self.stored_states[state_name] = []
                self.stored_states[state_name].append(result)
                # print(f"Captured state {state_name}, shape: {result.shape}")
            
            return result
        
        # Replace the forward method
        import types
        mixer.forward = types.MethodType(patched_forward, mixer)
        print(f"Installed hook for layer {self.target_layer} to capture {self.state_key}")
    
    def restore_hooks(self):
        """Restore original forward methods"""
        if self.original_mixer_forward is not None:
            blocks = self.model.blocks.layers
            target_block = blocks[self.target_layer]
            target_block.mixer.forward = self.original_mixer_forward
            print(f"Restored original forward method for layer {self.target_layer}")
    
    def forward(self, pts):
        """Forward pass with state extraction"""
        # Clear stored states
        self.stored_states = {}
        
        # Call the model's forward method
        output = self.model(pts)
        return output
    
    def get_loss_acc(self, ret, label):
        """Pass through to the model's get_loss_acc method"""
        return self.model.get_loss_acc(ret, label)
    
    def __del__(self):
        """Clean up hooks when the object is deleted"""
        self.restore_hooks()



def calculate_internal_correlation(model, points, target_class, state_key):
    """Calculate internal state correlation for a specific input and target class
    Implements Grad-ISC method from the paper (Equation 9)"""
    model.eval()
    
    # Access the right model instance (depending on whether we're using DataParallel)
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Clear stored states
    actual_model.stored_states = {}
    
    # Forward pass to capture states
    points = points.clone().detach().requires_grad_(True)
    outputs = model(points)
    
    # Get the prediction for target class
    target_score = outputs[0, target_class]
    
    # Check if state was captured
    if state_key not in actual_model.stored_states:
        print(f"Warning: State {state_key} not found in stored states!")
        available_keys = list(actual_model.stored_states.keys())
        if available_keys:
            print(f"Using {available_keys[0]} instead")
            state_key = available_keys[0]
        else:
            print("No states available")
            return torch.zeros(384, device=points.device)
    
    # Get the captured state
    state = actual_model.stored_states[state_key][0].clone()
    
    # We'll use a direct approach to compute gradients
    proxy_state = state.clone().detach().requires_grad_(True)
    
    # Create a computational path from state to output
    proxy_output = (proxy_state * state).sum() / (state * state).sum() * target_score
    
    # Compute gradient of proxy output with respect to proxy state
    proxy_output.backward(retain_graph=True)
    
    # Get the gradient
    state_grad = proxy_state.grad
    
    if state_grad is None:
        print("Warning: Gradient computation failed! Using approximation.")
        # Fallback method using input gradients
        target_score.backward(retain_graph=True)
        input_grad = points.grad
        
        # Use a simplified approximation
        state_grad = torch.ones_like(state)
    
    # Calculate internal correlation as element-wise product of state and gradient
    # This is i^(ℓ,x)_n = g^(ℓ,x)_n ⊙ x^(ℓ)_n in the paper
    correlation = (state_grad * state).detach()
    
    # If the state has multiple dimensions, flatten for feature importance
    if len(correlation.shape) > 1:
        correlation = correlation.reshape(-1)
    
    return correlation


def create_class_templates(model, dataloader, num_classes=40, samples_per_class=15, state_key="s_11", logger=None):
    """Create internal correlation templates for each class
    Implements the template creation from Definition 2 in the paper"""
    print_log(f"Creating class templates using state {state_key}...", logger=logger)
    
    # Dictionary to store correctly classified samples by class
    class_samples = {i: [] for i in range(num_classes)}
    class_count = {i: 0 for i in range(num_classes)}
    
    # Find correctly classified samples (process in batches)
    model.eval()
    
    # We'll process classes one at a time
    templates = {}
    
    with torch.no_grad():
        # First pass: Identify and collect sample indices for correctly classified samples
        sample_indices = {i: [] for i in range(num_classes)}
        
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataloader, desc="Finding sample indices")):
            points = data[0].cuda()
            labels = data[1].cuda()
            
            # Skip if we have enough samples for all classes
            if all(len(sample_indices[i]) >= samples_per_class for i in range(num_classes)):
                break
            
            # Get predictions
            outputs = model(points)
            preds = outputs.argmax(dim=1)
            
            # Store correctly classified sample indices
            for i, (pred, label) in enumerate(zip(preds, labels)):
                label_item = label.item()
                if pred == label and len(sample_indices[label_item]) < samples_per_class:
                    sample_indices[label_item].append((idx, i))
    
    # Second pass: Process one class at a time to save memory
    for class_idx in range(num_classes):
        print_log(f"Processing class {class_idx}, samples: {len(sample_indices[class_idx])}", logger=logger)
        
        if not sample_indices[class_idx]:
            print_log(f"No samples found for class {class_idx}, skipping", logger=logger)
            continue
        
        # Collect samples for this class in batches of 5
        batch_size = 5
        all_correlations = []
        
        for i in range(0, len(sample_indices[class_idx]), batch_size):
            batch_indices = sample_indices[class_idx][i:i+batch_size]
            
            # Process each sample in the batch
            for batch_idx, sample_idx in batch_indices:
                # Reload the sample
                for current_idx, (_, _, data) in enumerate(dataloader):
                    if current_idx == batch_idx:
                        points = data[0][sample_idx:sample_idx+1].cuda()
                        break
                
                # Calculate correlation for this sample
                correlation = calculate_internal_correlation(model, points, class_idx, state_key)
                all_correlations.append(correlation)
                
                # Clean up to save memory
                del points
                torch.cuda.empty_cache()
        
        # Only continue if we have samples
        if not all_correlations:
            continue
        
        # Stack all correlations for this class
        try:
            correlations_tensor = torch.stack(all_correlations).cuda()
            
            # Calculate threshold for binarization (beta in the paper)
            # beta = torch.mean(correlations_tensor)
            beta = 0.5
            
            # Binarize correlations
            binary_correlations = (correlations_tensor > beta).float()
            
            # Calculate simplicity part - ED(1/j * Σ i^(ℓ,x)+_n,j)
            simplicity = torch.mean(binary_correlations, dim=0)
            
            # Calculate homogeneity part
            J = binary_correlations.shape[0]
            if J > 1:
                # Initialize XOR sum
                xor_sum = torch.zeros_like(simplicity)
                
                # Process in smaller chunks to save memory
                chunk_size = min(J, 5)
                for i in range(0, J, chunk_size):
                    i_end = min(i + chunk_size, J)
                    chunk_i = binary_correlations[i:i_end]
                    
                    for j in range(J):
                        if j not in range(i, i_end):  # Avoid comparing with itself
                            chunk_j = binary_correlations[j:j+1]
                            # XOR between binary vectors is equivalent to (a != b)
                            xor_result = (chunk_i != chunk_j).float()
                            xor_sum += xor_result.sum(dim=0)
                
                # Normalize by number of pairs
                num_pairs = (J * (J - 1)) // 2  # Integer division
                homogeneity = 1.0 - (xor_sum / (2 * num_pairs))  # Divide by 2 because we double-counted
            else:
                # If only one sample, homogeneity is perfect
                homogeneity = torch.ones_like(simplicity)
            
            # Calculate ICS score according to Equation 10
            ics_score = simplicity * homogeneity
            
            # Store template
            templates[class_idx] = ics_score.cpu()
            
            # Clean up
            del correlations_tensor, binary_correlations, simplicity, homogeneity, ics_score
            torch.cuda.empty_cache()
            
        except Exception as e:
            print_log(f"Error creating template for class {class_idx}: {e}", logger=logger)
            continue
    
    # Convert CPU templates back to GPU for further processing
    for class_idx in templates:
        templates[class_idx] = templates[class_idx].cuda()
    
    return templates


def visualize_internal_correlations(templates, output_dir='visualizations', logger=None):
    """Visualize internal correlation templates"""
    
    os.makedirs(output_dir, exist_ok=True)
    print_log(f"Created visualization directory at {output_dir}", logger=logger)
    
    # Move templates to CPU for visualization
    cpu_templates = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in templates.items()}
    
    # Visualize each class template
    for class_idx, template in cpu_templates.items():
        plt.figure(figsize=(15, 10))
        
        # Reshape for better visualization
        vis_template = template.reshape(1, -1)
        
        # Create heatmap of the template
        sns.heatmap(vis_template, cmap='viridis', cbar=True,
                   xticklabels=False, yticklabels=False)
        
        plt.title(f"Class {class_idx} Internal Correlation Template")
        plt.tight_layout()
        plt.xlabel('Feature Dimensions')
        plt.ylabel("Sample")
        plt.savefig(f"{output_dir}/class_{class_idx}_template.png")
        plt.close()
    
    # Create a similarity matrix between templates
    if len(cpu_templates) > 1:
        similarity_matrix = np.zeros((len(cpu_templates), len(cpu_templates)))
        class_indices = list(cpu_templates.keys())
        
        for i, idx1 in enumerate(class_indices):
            for j, idx2 in enumerate(class_indices):
                # Calculate cosine similarity
                template1 = cpu_templates[idx1].flatten()
                template2 = cpu_templates[idx2].flatten()
                dot_product = np.dot(template1, template2)
                norm1 = np.linalg.norm(template1)
                norm2 = np.linalg.norm(template2)
                similarity = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
                similarity_matrix[i, j] = similarity
        
        # Visualize similarity matrix
        plt.figure(figsize=(15, 15))
        sns.heatmap(similarity_matrix, cmap='coolwarm', annot=True, fmt=".2f",
                   xticklabels=class_indices, yticklabels=class_indices)
        plt.title("Template Similarity Between Classes")
        plt.tight_layout()
        plt.xlabel('Class Index')
        plt.ylabel('Class Index')
        plt.savefig(f"{output_dir}/template_similarity_matrix.png")
        plt.close()

def internal_repair_loss(model, inputs, targets, templates, state_key="s_11", gamma=1e-1):
    """Calculate internal repair loss for difficult samples
    Implements the repair method from Section 4.2 of the paper (Equation 13 and 14)"""
    batch_size = inputs.size(0)
    
    # Access the right model instance
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Forward pass to get predictions
    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
    
    # Identify difficult samples (incorrect predictions)
    difficult_indices = [i for i in range(batch_size) if preds[i] != targets[i] and targets[i].item() in templates]
    
    if not difficult_indices:
        return torch.tensor(0.0, device=inputs.device)
    
    # For memory efficiency, process samples in smaller batches
    max_batch_size = 5
    total_loss = torch.tensor(0.0, device=inputs.device)
    
    for i in range(0, len(difficult_indices), max_batch_size):
        batch_indices = difficult_indices[i:i+max_batch_size]
        
        # Process this batch
        batch_samples = inputs[batch_indices].clone().detach().requires_grad_(True)
        batch_targets = targets[batch_indices]
        
        # Clear stored states
        actual_model.stored_states = {}
        
        # Forward pass
        model.eval()  # Switch to eval mode to handle batch norm
        outputs = model(batch_samples)
        model.train()  # Switch back to train mode
        
        # Check if we captured the state
        if state_key not in actual_model.stored_states:
            print(f"Warning: State {state_key} not found in stored states!")
            if actual_model.stored_states:
                available_keys = list(actual_model.stored_states.keys())
                print(f"Available states: {available_keys}")
                state_key = available_keys[0]
            else:
                print("No states captured!")
                continue
        
        # Get the captured state
        state = actual_model.stored_states[state_key][0]
        
        # Process each sample in the batch
        batch_loss = torch.tensor(0.0, device=inputs.device)
        for j, (idx, target) in enumerate(zip(batch_indices, batch_targets)):
            target_class = target.item()
            
            if target_class not in templates:
                continue
                
            template = templates[target_class]
            
            # Extract this sample's state
            if state.dim() > 2:  # If state has batch dimension
                sample_state = state[j]
            else:
                sample_state = state
            
            # Flatten the state
            state_flat = sample_state.reshape(-1)
            
            # Ensure dimensions match
            if state_flat.shape[0] != template.shape[0]:
                min_size = min(state_flat.shape[0], template.shape[0])
                state_flat = state_flat[:min_size]
                template = template[:min_size]
            
            # Calculate the dot product as in the paper
            # ED(i^(ℓ,x)_n ⊙ î^(ℓ,s)+_n)
            inverted_template = 1.0 - template
            loss = torch.mean(state_flat * inverted_template)
            batch_loss += loss
        
        # Add batch loss to total
        if len(batch_indices) > 0:
            total_loss += batch_loss
        
        # Clean up
        del batch_samples, batch_targets, state
        torch.cuda.empty_cache()
    
    # Calculate the final loss with gamma weighting
    if difficult_indices:
        repair_loss = gamma * total_loss / len(difficult_indices)
    else:
        repair_loss = torch.tensor(0.0, device=inputs.device)
    
    return repair_loss

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # Build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    
    # Build model
    base_model = builder.model_builder(config.model)

    # Parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # Resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
            print_log(f'Loaded model from {args.ckpts}', logger=logger)
        else:
            print_log('Training from scratch', logger=logger)

    # Create visualization directory
    vis_dir = os.path.join(args.experiment_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Wrap model with PointMambaWithStates to access internal states
    wrapped_model = PointMambaWithStates(base_model, state_key=args.state_key)
    
    # Move model to GPU and apply DDP if needed
    if args.use_gpu:
        wrapped_model.to(args.local_rank)
    
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            wrapped_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wrapped_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        wrapped_model = nn.parallel.DistributedDataParallel(wrapped_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        wrapped_model = nn.DataParallel(wrapped_model).cuda()
    
    
    # Evaluate model before repair
    print_log("Evaluating model before repair...", logger=logger)
    pre_metrics = validate(wrapped_model, test_dataloader, 0, val_writer, args, config, logger=logger)
    pre_accuracy = pre_metrics.acc
    
    # Create templates for each class for the internal repair
    print_log(f"Creating class templates with {args.samples_per_class} samples per class...", logger=logger)
    templates = create_class_templates(
        wrapped_model.module if args.distributed else wrapped_model,
        train_dataloader,
        num_classes=config.model.cls_dim,
        samples_per_class=args.samples_per_class,
        state_key=args.state_key,
        logger=logger
    )
    
    print('saving visualisations')
    # Visualize the templates
    visualize_internal_correlations(
        templates, 
        output_dir=vis_dir,
        logger=logger
    )
    print('visualisations saved')
    
    # Optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(wrapped_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # Training setup
    wrapped_model.zero_grad()
    misc.summary_parameters(wrapped_model, logger=logger)

    # Training loop
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        wrapped_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc', 'repair_loss'])
        num_iter = 0
        
        n_batches = len(train_dataloader)
        npoints = config.npoints

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                   desc=f"Epoch {epoch}/{config.max_epoch}")
        
        for idx, (taxonomy_ids, model_ids, data) in pbar:
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            
            if 'scan' in args.config:
                points = train_transforms(points)
            else:
                points = train_transforms_raw(points)
            
            # Forward pass
            ret = wrapped_model(points)
            
            # Standard loss
            ce_loss, acc = wrapped_model.module.get_loss_acc(ret, label)
            
            # Internal repair loss
            repair_loss = internal_repair_loss(
                wrapped_model.module if args.distributed else wrapped_model,
                points, label, templates, 
                state_key=args.state_key, 
                gamma=args.gamma
            )
            
            # Total loss
            loss = ce_loss + repair_loss
            
            loss.backward()
            
            # Gradient accumulation if needed
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                wrapped_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                repair_loss = dist_utils.reduce_tensor(repair_loss, args)
                losses.update([loss.item(), acc.item(), repair_loss.item()])
            else:
                losses.update([loss.item(), acc.item(), repair_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/RepairLoss', repair_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}",
                'repair': f"{repair_loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        # else:
            # scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/RepairLoss', losses.avg(2), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(wrapped_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            better = metrics.better_than(best_metrics)
            
            # Save checkpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(wrapped_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                      logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
                
                # Also save as mender-best to distinguish from regular training
                builder.save_checkpoint(wrapped_model, optimizer, epoch, metrics, best_metrics, 'mender-best', args,
                                      logger=logger)
                
        # Save last checkpoint
        builder.save_checkpoint(wrapped_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                              logger=logger)
        
    # Calculate improvement
    print_log(f"Accuracy before repair: {pre_accuracy:.2f}%", logger=logger)
    print_log(f"Best accuracy after repair: {best_metrics.acc:.2f}%", logger=logger)
    print_log(f"Improvement: {best_metrics.acc - pre_accuracy:.2f}%", logger=logger)
    
    # Clean up hooks before exiting
    if args.distributed:
        wrapped_model.module.remove_hooks()
    # else:
        # wrapped_model.remove_hooks()
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)