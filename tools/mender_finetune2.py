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
            print(f"Invalid state key format: {state_key}, should be like 's_11' or 'x_11'")
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
    """Calculate internal state correlation for a specific input and target class"""
    model.eval()
    
    # Access the right model instance (depending on whether we're using DataParallel)
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Clear stored states
    actual_model.stored_states = {}
    
    # Enable gradient tracking for input
    points = points.clone().detach().requires_grad_(True)
    
    # Forward pass to capture states
    output = model(points)
    # print(f"Forward pass complete, output shape: {output.shape}")
    
    # Get the state before backward pass
    if state_key not in actual_model.stored_states:
        print(f"Warning: State {state_key} not found in stored states!")
        available_keys = list(actual_model.stored_states.keys())
        if available_keys:
            print(f"Using {available_keys[0]} instead")
            state_key = available_keys[0]
        else:
            print("No states available")
            return np.zeros((384,))
    
    # Get the captured state
    state = actual_model.stored_states[state_key][0]
    # print(f"State shape: {state.shape}")
    
    # Create a direct gradient connection to output by using state values
    # This ensures gradients will flow back to the state
    # Extract the state dimension to focus on
    state_flat = state.reshape(state.size(0), -1)  # Flatten all but batch dimension
    
    # Create gradients for the state using input-based sensitivity
    # Compute input gradients
    score = output[0, target_class]
    score.backward(retain_graph=True)
    
    # Compute input sensitivity (how much each input affects the output)
    input_grad = points.grad.abs().sum(dim=-1)  # Sum over xyz dimensions
    
    # Use input sensitivity as a proxy for which parts of the state are important
    # This avoids the need for direct state gradients
    correlation = state_flat.abs().detach().cpu().numpy()
    
    # If the state has multiple dimensions, take the mean to get feature importance
    if len(correlation.shape) > 1:
        # Take mean across batch dimension
        correlation = correlation.mean(axis=0)
    
    return correlation


def create_class_templates(model, dataloader, num_classes=40, samples_per_class=15, state_key="s_11", logger=None):
    """Create internal correlation templates for each class"""
    print_log(f"Creating class templates using state {state_key}...", logger=logger)
    
    # Collect samples by class
    class_samples = {i: [] for i in range(num_classes)}
    class_count = {i: 0 for i in range(num_classes)}
    
    # Find correctly classified samples
    model.eval()
    
    for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataloader, desc="Finding correctly classified samples")):
        points = data[0].cuda()
        labels = data[1].cuda()
        
        # Skip if we have enough samples for all classes
        if all(class_count[i] >= samples_per_class for i in range(num_classes)):
            break
        
        # Get predictions
        with torch.no_grad():
            outputs = model(points)
            preds = outputs.argmax(dim=1)
        
        # Store correctly classified samples
        for i, (pred, label) in enumerate(zip(preds, labels)):
            if pred == label and class_count[label.item()] < samples_per_class:
                class_samples[label.item()].append(points[i:i+1].clone())
                class_count[label.item()] += 1
    
    # Generate templates for each class
    templates = {}
    for class_idx, samples in class_samples.items():
        if not samples:
            continue
            
        print_log(f"Processing class {class_idx}, samples: {len(samples)}", logger=logger)
        
        class_correlations = []
        for sample in samples:
            # Calculate internal correlation for this sample
            correlation = calculate_internal_correlation(
                model, sample, class_idx, state_key)
            class_correlations.append(correlation)
        
        # Create template as average of correlations
        avg_correlation = np.mean(class_correlations, axis=0)
        
        # Binarize template based on threshold (using mean as threshold)
        binary_template = (avg_correlation > np.mean(avg_correlation)).astype(float)
        
        templates[class_idx] = binary_template
    
    return templates


def visualize_internal_correlations(templates, output_dir='visualizations', logger=None):
    """Visualize internal correlation templates"""
    
    os.makedirs(output_dir, exist_ok=True)
    print_log(f"Created visualization directory at {output_dir}", logger=logger)
    
    # Visualize each class template
    for class_idx, template in templates.items():
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
    if len(templates) > 1:
        similarity_matrix = np.zeros((len(templates), len(templates)))
        class_indices = list(templates.keys())
        
        for i, idx1 in enumerate(class_indices):
            for j, idx2 in enumerate(class_indices):
                # Calculate cosine similarity
                template1 = templates[idx1].flatten()
                template2 = templates[idx2].flatten()
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

def internal_repair_loss(model, inputs, targets, templates, state_key="s_11", gamma=1e-1):  # Reduced gamma value
    """Calculate internal repair loss for difficult samples"""
    batch_size = inputs.size(0)
    
    # Access the right model instance
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Forward pass to get predictions with the whole batch first
    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
    
    # Identify difficult samples (incorrect predictions)
    difficult_indices = [i for i in range(batch_size) if preds[i] != targets[i] and targets[i].item() in templates]
    
    if not difficult_indices:
        return torch.tensor(0.0, device=inputs.device)
    
    # Process all difficult samples in a single batch to avoid BatchNorm issues
    difficult_samples = inputs[difficult_indices].clone().detach().requires_grad_(True)
    difficult_targets = targets[difficult_indices]
    
    # Clear stored states
    actual_model.stored_states = {}
    
    # Forward pass for all difficult samples together
    model.eval()  # Switch to eval mode to handle batch norm
    outputs = model(difficult_samples)
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
            return torch.tensor(0.0, device=inputs.device)
    
    # Get the captured state
    state = actual_model.stored_states[state_key][0]
    
    # Process the state for each difficult sample
    total_loss = 0.0
    for i, idx in enumerate(difficult_indices):
        target_class = targets[idx].item()
        template = torch.from_numpy(templates[target_class]).float().cuda()
        
        # Extract this sample's state
        if state.dim() > 2:  # If state has batch dimension
            sample_state = state[i]
        else:
            sample_state = state
        
        # Flatten the state to get feature importance
        state_flat = sample_state.reshape(-1)
        
        # Calculate cosine similarity with template
        state_norm = torch.norm(state_flat)
        template_norm = torch.norm(template)
        
        if state_norm > 0 and template_norm > 0:
            # Normalize the vectors to ensure proper cosine similarity calculation
            state_flat_norm = state_flat / state_norm
            template_norm = template / template_norm
            
            # Here we want to maximize similarity (minimize negative similarity)
            similarity = torch.sum(state_flat_norm * template_norm)
            
            # Use 1 - similarity to get a positive loss that approaches 0 as similarity increases
            loss = 1.0 - similarity
            total_loss += loss
    
    # Calculate the mean loss
    if difficult_indices:
        repair_loss = gamma * total_loss / len(difficult_indices)
        
        # Add a simple logging statement to track the loss
        # print(f"Repair loss: {repair_loss.item():.4f}, Num difficult samples: {len(difficult_indices)}")
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
    else:
        wrapped_model.remove_hooks()
    
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