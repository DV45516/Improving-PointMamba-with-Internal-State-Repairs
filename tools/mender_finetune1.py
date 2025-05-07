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


# class PointMambaWithStates(nn.Module):
#     """Wrapper for PointMamba that extracts internal states"""
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.state_hooks = []
#         self.stored_states = {}
#         self.hook_blocks()
        
#     # def hook_blocks(self):
#     #     """Register hooks to capture internal states from the Mamba blocks"""
#     #     # Clear existing hooks
#     #     for hook in self.state_hooks:
#     #         hook.remove()
#     #     self.state_hooks = []
        
#     #     # Register hooks for states in the PointMamba blocks
#     #     for i, block in enumerate(self.model.blocks.layers):
#     #         # Hook for state x (linear mapping output)
#     #         def get_hook_fn_x(block_idx):
#     #             def hook_fn(module, input, output):
#     #                 state_key = f"x_{block_idx}"
#     #                 if state_key not in self.stored_states:
#     #                     self.stored_states[state_key] = []
#     #                 self.stored_states[state_key].append(output)
#     #             return hook_fn
            
#     #         # Register the hook
#     #         if hasattr(block, 'mixer') and hasattr(block.mixer, 'in_proj'):
#     #             hook = block.mixer.in_proj.register_forward_hook(get_hook_fn_x(i))
#     #             self.state_hooks.append(hook)
    
#     def hook_blocks(self):
#         """Register hooks to capture internal states from the Mamba blocks"""
#         # Clear existing hooks
#         for hook in self.state_hooks:
#             hook.remove()
#         self.state_hooks = []
        
#         # Debug the model structure first
#         print("Debugging model structure...")
        
#         # Check if blocks exist
#         if not hasattr(self.model, 'blocks'):
#             print("Model does not have 'blocks' attribute!")
#             return
            
#         # Check if layers exist in blocks
#         if not hasattr(self.model.blocks, 'layers'):
#             print("Model.blocks does not have 'layers' attribute!")
#             return
        
#         print(f"Number of blocks: {len(self.model.blocks.layers)}")
        
#         # Register hooks for multiple possible locations
#         for i, block in enumerate(self.model.blocks.layers):
#             print(f"Block {i} attributes: {dir(block)}")
            
#             # Different possible hook locations
#             if hasattr(block, 'mixer') and hasattr(block.mixer, 'in_proj'):
#                 print(f"Found mixer.in_proj in block {i}")
                
#                 def get_hook_fn_x(block_idx):
#                     def hook_fn(module, input, output):
#                         state_key = f"x_{block_idx}"
#                         if state_key not in self.stored_states:
#                             self.stored_states[state_key] = []
#                         self.stored_states[state_key].append(output)
#                     return hook_fn
                
#                 # Register hook
#                 hook = block.mixer.in_proj.register_forward_hook(get_hook_fn_x(i))
#                 self.state_hooks.append(hook)
#                 print(f"Registered hook for x_{i}")
            
#             # Try alternative paths
#             elif hasattr(block, 'norm1') and hasattr(block, 'mixer'):
#                 print(f"Found alternative structure in block {i}")
#                 print(f"Mixer attributes: {dir(block.mixer)}")
                
#                 # Try to find a suitable component to hook
#                 if hasattr(block.mixer, 'to_qkv'):
#                     hook = block.mixer.to_qkv.register_forward_hook(get_hook_fn_x(i))
#                     self.state_hooks.append(hook)
#                     print(f"Registered hook for x_{i} on mixer.to_qkv")
#                 elif hasattr(block.mixer, 'linear'):
#                     hook = block.mixer.linear.register_forward_hook(get_hook_fn_x(i))
#                     self.state_hooks.append(hook)
#                     print(f"Registered hook for x_{i} on mixer.linear")
    
#     def forward(self, pts):
#         """Forward pass with state extraction"""
#         # Clear stored states
#         self.stored_states = {}
        
#         # Call the model's forward method
#         output = self.model(pts)
        
#         return output
    
#     def forward_with_states(self, pts):
#         """Forward pass with state extraction"""
#         # Clear stored states
#         self.stored_states = {}
        
#         # Call the model's forward method
#         output = self.model(pts)
        
#         return output, self.stored_states
    
#     def get_loss_acc(self, ret, label):
#         """Pass through to the model's get_loss_acc method"""
#         return self.model.get_loss_acc(ret, label)
    
#     def remove_hooks(self):
#         """Remove all hooks to prevent memory leaks"""
#         for hook in self.state_hooks:
#             hook.remove()
#         self.state_hooks = []


# def calculate_internal_correlation(model, points, target_class, state_key):
#     """Calculate internal state correlation for a specific input and target class"""
#     model.eval()
    
#     # Enable gradient tracking
#     points.requires_grad_(True)
    
#     # Forward pass with state extraction
#     output, states = model.forward_with_states(points)
    
#     # Calculate gradient w.r.t target class
#     model.zero_grad()
#     score = output[0, target_class]
#     score.backward(retain_graph=True)
    
#     # Get the internal state and its gradient
#     if state_key not in states or not states[state_key]:
#         print(f"Warning: State {state_key} not found in stored states!")
#         return np.zeros((384,))  # Return zeros with expected dimensions
    
#     state = states[state_key][0]
    
#     # Calculate correlation (element-wise product of state and its gradient)
#     if state.grad is None:
#         print(f"Warning: Gradient for state {state_key} is None!")
#         return np.zeros((384,))
    
#     correlation = state * state.grad
    
#     # Detach and convert to numpy
#     correlation = correlation.detach().cpu().numpy()
    
#     # Return the mean along all dimensions except the feature dimension
#     if len(correlation.shape) > 1:
#         correlation = correlation.mean(tuple(range(len(correlation.shape)-1)))
    
#     return correlation

class PointMambaWithStates(nn.Module):
    """Wrapper for PointMamba that extracts internal states"""
    def __init__(self, model, state_key="x_11"):
        super().__init__()
        self.model = model
        self.state_hooks = []
        self.stored_states = {}
        self.state_key = state_key
        self.hook_blocks()
        
    def hook_blocks(self):
        """Register hooks to capture internal states from the Mamba blocks"""
        # Clear existing hooks
        for hook in self.state_hooks:
            hook.remove()
        self.state_hooks = []
        
        # Debug mode to see the model structure
        debug_mode = False
        
        # print('eneterd')
        
        if debug_mode:
            print("Number of blocks:", len(self.model.blocks.layers))
            # Print structure of first block to understand the architecture
            if len(self.model.blocks.layers) > 0:
                print("Block 0 attributes:", dir(self.model.blocks.layers[0]))
                if hasattr(self.model.blocks.layers[0], 'mixer'):
                    print("Mixer attributes:", dir(self.model.blocks.layers[0].mixer))
        
        # print(self.model.blocks.layers)
        
        # Try to hook each block's in_proj and save state
        for i, block in enumerate(self.model.blocks.layers):
            # Check if accessing a specific state (e.g., x_11)
            block_idx = int(self.state_key.split('_')[1]) if '_' in self.state_key else None
            
            
            
            # Only add a hook if this is the block we want to analyze
            # or if we're monitoring all blocks (block_idx is None)
            if block_idx is not None and i != block_idx:
                continue
            
            # print(i, block_idx)
               
            if hasattr(block, 'mixer') and hasattr(block.mixer, 'in_proj'):
                # Define a hook function to capture the state
                def get_hook_fn(idx):
                    def hook_fn(module, input, output):
                        state_name = f"x_{idx}"
                        if state_name not in self.stored_states:
                            self.stored_states[state_name] = []
                        self.stored_states[state_name].append(output)
                        print(self.stored_states)
                        if debug_mode:
                            print('\n')
                            print(f"Captured state {state_name}, shape: {output.shape}")
                            print('\n')
                    return hook_fn
                
                # Register the hook
                hook = block.mixer.in_proj.register_forward_hook(get_hook_fn(i))
                # print(hook)
                self.state_hooks.append(hook)
                if debug_mode:
                    print(f"Registered hook for block {i}")

        # print('\n stored states: \n')
        # print(self.stored_states)
        
    def forward(self, pts):
        """Forward pass with state extraction"""
        # Clear stored states
        self.stored_states = {}
        
        # Call the model's forward method
        output = self.model(pts)
        
        print("DEBUG: After forward pass")
        print("DEBUG: stored_states:", self.stored_states.keys())
        
        return output
    
    def get_loss_acc(self, ret, label):
        """Pass through to the model's get_loss_acc method"""
        return self.model.get_loss_acc(ret, label)
    
    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks"""
        for hook in self.state_hooks:
            hook.remove()
        self.state_hooks = []

# def calculate_internal_correlation(model, points, target_class, state_key):
#     """Calculate internal state correlation for a specific input and target class"""
#     model.eval()
    
#     # Enable gradient tracking
#     points.requires_grad_(True)
    
#     # Clear stored states
#     if hasattr(model, 'stored_states'):
#         model.stored_states = {}
#     elif hasattr(model, 'module') and hasattr(model.module, 'stored_states'):
#         model.module.stored_states = {}
    
#     # Forward pass with state extraction
#     model.zero_grad()
#     output = model(points)
    
#     # Calculate gradient w.r.t target class
#     score = output[0, target_class]
#     score.backward(retain_graph=True)
    
#     # Get the internal state and its gradient
#     # Access the stored states correctly depending on whether we're using DataParallel
#     if hasattr(model, 'stored_states'):
#         states = model.stored_states
#     elif hasattr(model, 'module') and hasattr(model.module, 'stored_states'):
#         states = model.module.stored_states
#     else:
#         print(f"Warning: Cannot find stored_states in model or model.module!")
#         return np.zeros((384,))  # Return zeros with expected dimensions
    
#     if state_key not in states or not states[state_key]:
#         print(f"Warning: State {state_key} not found in stored states!")
#         return np.zeros((384,))  # Return zeros with expected dimensions
    
#     state = states[state_key][0]
    
#     # Check if gradient exists
#     if state.grad is None:
#         print(f"Warning: Gradient for state {state_key} is None!")
#         return np.zeros((384,))
    
#     # Calculate correlation (element-wise product of state and its gradient)
#     correlation = state * state.grad
    
#     # Detach and convert to numpy
#     correlation = correlation.detach().cpu().numpy()
    
#     # Return the mean along all dimensions except the feature dimension
#     if len(correlation.shape) > 1:
#         correlation = correlation.mean(tuple(range(len(correlation.shape)-1)))
    
#     return correlation


def calculate_internal_correlation(model, points, target_class, state_key):
    """Calculate internal state correlation for a specific input and target class"""
    model.eval()
    
    # Enable gradient tracking
    points.requires_grad_(True)
    
    
    
    # Clear stored states
    if hasattr(model, 'stored_states'):
        model.stored_states = {}
    elif hasattr(model, 'module') and hasattr(model.module, 'stored_states'):
        model.module.stored_states = {}
    
    # Forward pass to capture states and prediction
    model.zero_grad()
    output = model(points)
    
    # Calculate gradient w.r.t target class
    score = output[0, target_class]
    score.backward(retain_graph=True)
    
    # Access the stored states correctly depending on whether we're using DataParallel
    if hasattr(model, 'stored_states'):
        states = model.stored_states
    elif hasattr(model, 'module') and hasattr(model.module, 'stored_states'):
        print(model.module.stored_states)
        states = model.module.stored_states
    else:
        print(f"Warning: Cannot find stored_states in model or model.module!")
        return np.zeros((384,))  # Return zeros with expected dimensions
    
    # print(states)
    
    # Check if the desired state was captured
    if state_key not in states or not states[state_key]:
        print(f"Warning: State {state_key} not found in stored states!")
        
        # Instead of returning zeros, try to use any available state
        available_keys = list(states.keys())
        if available_keys:
            print(f"Available states: {available_keys}, using {available_keys[0]} instead")
            state_key = available_keys[0]
        else:
            # raise(ValueError) 
            # print('No state available')
            return np.zeros((384,))  # Return zeros if no states available
    
    # Get the state and its gradient
    state = states[state_key][0]
    
    # Check if gradient exists
    if state.grad is None:
        print(f"Warning: Gradient for state {state_key} is None!")
        return np.zeros((384,))
    
    # Calculate correlation (element-wise product of state and its gradient)
    correlation = state * state.grad
    
    # Detach and convert to numpy
    correlation = correlation.detach().cpu().numpy()
    
    # Handle multi-dimensional states by taking the mean across all dimensions except the last
    if len(correlation.shape) > 1:
        # Keep only the feature dimension (last dimension)
        reduction_dims = tuple(range(len(correlation.shape)-1))
        if reduction_dims:
            correlation = correlation.mean(reduction_dims)
    
    return correlation

def create_class_templates(model, dataloader, num_classes=40, samples_per_class=15, state_key="x_11", logger=None):
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print_log(f"Created visualization directory at {output_dir}", logger=logger)
    
    # Visualize each class template
    for class_idx, template in templates.items():
        plt.figure(figsize=(10, 6))
        
        # Reshape for better visualization
        vis_template = template.reshape(1, -1)
        
        # Create heatmap of the template
        sns.heatmap(vis_template, cmap='viridis', cbar=True,
                   xticklabels=False, yticklabels=False)
        
        plt.title(f"Class {class_idx} Internal Correlation Template")
        plt.tight_layout()
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
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, cmap='coolwarm', annot=True, fmt=".2f",
                   xticklabels=class_indices, yticklabels=class_indices)
        plt.title("Template Similarity Between Classes")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/template_similarity_matrix.png")
        plt.close()


def internal_repair_loss(model, inputs, targets, templates, state_key="x_11", gamma=1e7):
    """Calculate internal repair loss for difficult samples"""
    batch_size = inputs.size(0)
    repair_loss = 0.0
    
    # Forward pass to get predictions
    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
    
    # Process each sample in the batch
    difficult_count = 0
    for i in range(batch_size):
        # Skip if prediction is correct
        if preds[i] == targets[i]:
            continue
        
        # Skip if we don't have a template for this class
        target = targets[i].item()
        if target not in templates:
            continue
        
        # Get template for this class
        template = torch.from_numpy(templates[target]).float().cuda()
        
        # Process this single difficult sample
        difficult_count += 1
        sample = inputs[i:i+1].clone().detach().requires_grad_(True)
        
        # Calculate internal correlation
        correlation = calculate_internal_correlation(model, sample, target, state_key)
        correlation_tensor = torch.from_numpy(correlation).float().cuda()
        
        # Calculate consistency with template (maximize consistency)
        consistency = -torch.mean(correlation_tensor * template)
        repair_loss += consistency
    
    # Return average loss (or zero if no difficult samples)
    if difficult_count > 0:
        return gamma * repair_loss / difficult_count
    else:
        return torch.tensor(0.0, device=inputs.device)


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
    
    # # Wrap model with PointMambaWithStates to access internal states
    # wrapped_model = PointMambaWithStates(base_model)
    
    # Wrap model with PointMambaWithStates to access internal states
    wrapped_model = PointMambaWithStates(base_model, state_key=args.state_key)
    
    # print('\n model:\n')
    # print(wrapped_model.stored_states)
    
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
    
    # wrapped_model = wrapped_model.cuda()
    
    # Evaluate model before repair
    print_log("Evaluating model before repair...", logger=logger)
    pre_metrics = validate(wrapped_model, test_dataloader, 0, val_writer, args, config, logger=logger)
    pre_accuracy = pre_metrics.acc
    
    # # Create templates for each class for the internal repair
    # print_log(f"Creating class templates with {args.samples_per_class} samples per class...", logger=logger)
    # templates = create_class_templates(
    #     wrapped_model.module if args.distributed else wrapped_model,
    #     train_dataloader,
    #     num_classes=config.model.cls_dim,
    #     samples_per_class=args.samples_per_class,
    #     state_key=args.state_key,
    #     logger=logger
    # )
    
    print_log(f"Creating class templates with {args.samples_per_class} samples per class for state {args.state_key}...", logger=logger)
    templates = create_class_templates(
        wrapped_model.module if args.distributed else wrapped_model,
        train_dataloader,
        num_classes=config.model.cls_dim,
        samples_per_class=args.samples_per_class,
        state_key=args.state_key,
        logger=logger
    )
    
    # Visualize the templates
    visualize_internal_correlations(
        templates, 
        output_dir=vis_dir,
        logger=logger
    )
    
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
        else:
            scheduler.step(epoch)
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