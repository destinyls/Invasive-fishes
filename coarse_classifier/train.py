import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset import FishDataset, ClassPairFishDataset
from model import FishClassifier
import math
import torch.profiler
import logging
import signal
import sys
import time
from datetime import datetime, timedelta

# Enable CUDNN benchmark for potential speedup if input sizes are constant
torch.backends.cudnn.benchmark = True

class TimeTracker:
    """Enhanced time tracking utility for training progress"""
    
    def __init__(self, total_epochs, batches_per_epoch):
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.total_steps = total_epochs * batches_per_epoch
        
        self.training_start_time = time.time()
        self.epoch_start_times = []
        self.step_start_time = None
        self.step_times = []
        self.epoch_times = []
        
        self.current_epoch = 0
        self.current_step_in_epoch = 0
        self.total_steps_completed = 0
        
    def start_epoch(self, epoch):
        """Start tracking a new epoch"""
        self.current_epoch = epoch
        self.current_step_in_epoch = 0
        self.epoch_start_times.append(time.time())
        
    def start_step(self):
        """Start tracking a new step/batch"""
        self.step_start_time = time.time()
        
    def complete_step(self):
        """Complete current step and record timing"""
        if self.step_start_time is not None:
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)
            self.current_step_in_epoch += 1
            self.total_steps_completed += 1
            
    def complete_epoch(self):
        """Complete current epoch and record timing"""
        if self.epoch_start_times:
            epoch_duration = time.time() - self.epoch_start_times[-1]
            self.epoch_times.append(epoch_duration)
            
    def get_step_progress_info(self):
        """Get progress information for current step"""
        if not self.step_times:
            return None
            
        # Calculate average step time
        recent_steps = min(50, len(self.step_times))  # Use last 50 steps for more accurate estimate
        avg_step_time = sum(self.step_times[-recent_steps:]) / recent_steps
        
        # Calculate remaining steps in current epoch
        remaining_steps_in_epoch = self.batches_per_epoch - self.current_step_in_epoch
        remaining_time_in_epoch = remaining_steps_in_epoch * avg_step_time
        
        # Calculate total remaining steps
        remaining_epochs = self.total_epochs - self.current_epoch
        remaining_steps_total = remaining_epochs * self.batches_per_epoch + remaining_steps_in_epoch
        remaining_time_total = remaining_steps_total * avg_step_time
        
        # Calculate current step duration
        current_step_duration = time.time() - self.step_start_time if self.step_start_time else 0
        
        return {
            'step_progress': f"{self.current_step_in_epoch + 1}/{self.batches_per_epoch}",
            'epoch_progress': f"{self.current_epoch + 1}/{self.total_epochs}",
            'total_progress': f"{self.total_steps_completed + 1}/{self.total_steps}",
            'current_step_duration': current_step_duration,
            'avg_step_time': avg_step_time,
            'remaining_time_in_epoch': remaining_time_in_epoch,
            'remaining_time_total': remaining_time_total,
            'completion_percentage': ((self.total_steps_completed + 1) / self.total_steps) * 100
        }
        
    def get_epoch_progress_info(self):
        """Get progress information for current epoch"""
        if not self.epoch_times:
            return None
            
        # Calculate average epoch time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        
        # Calculate remaining time
        remaining_epochs = self.total_epochs - (self.current_epoch + 1)
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        
        # Calculate total elapsed time
        total_elapsed = time.time() - self.training_start_time
        
        # Calculate ETA
        estimated_completion_time = self.training_start_time + total_elapsed + estimated_remaining_time
        
        # Get current epoch duration
        current_epoch_duration = time.time() - self.epoch_start_times[-1] if self.epoch_start_times else 0
        
        return {
            'current_epoch_duration': current_epoch_duration,
            'avg_epoch_time': avg_epoch_time,
            'total_elapsed': total_elapsed,
            'estimated_remaining_time': estimated_remaining_time,
            'eta': estimated_completion_time,
            'completion_percentage': ((self.current_epoch + 1) / self.total_epochs) * 100
        }
        
    @staticmethod
    def format_time(seconds):
        """Format seconds into human readable time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
        
    @staticmethod
    def format_eta(timestamp):
        """Format timestamp into ETA string"""
        return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

def setup_logger(rank):
    """Setup logger for distributed training"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def cleanup():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    print(f"Received signal {signum}, cleaning up...")
    cleanup()
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fish Classification Model")
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory with class subdirectories')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Path to validation data directory with class subdirectories')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (per GPU)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (if continuing training)')
    parser.add_argument('--save_path', type=str, default="models/fish_classifier.pth",
                        help="Path to save the trained model")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'efficientnet_b2'],
                        help='Backbone network architecture')
    parser.add_argument('--use_class_pairs', type=bool, default=True,
                        help='Use class-paired BYOL instead of standard BYOL')
    parser.add_argument('--profile_batches', type=int, default=0, 
                        help='Number of batches to profile per epoch (0 to disable). Profiling done on rank 0.')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='Local rank for distributed training (deprecated, use LOCAL_RANK env var)')
    
    return parser.parse_args()

def setup_distributed():
    """Setup distributed training environment"""
    # Get rank and world size from environment variables (torchrun sets these)
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if rank == -1 or local_rank == -1:
        # Not using distributed training
        return False, 0, 0, 1, torch.device("cuda"if torch.cuda.is_available() else "cpu")
    
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # Set the GPU device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return True, rank, local_rank, world_size, device

def create_data_loaders(args, is_distributed, rank, world_size):
    """Create distributed data loaders"""
    logger = logging.getLogger(__name__)
    
    if rank == 0:
        logger.info(f"Loading training data from {args.train_dir}")
    
    if args.use_class_pairs:
        if rank == 0:
            logger.info("Using class-paired BYOL approach with images from same class...")
        train_dataset, val_dataset = ClassPairFishDataset.get_datasets(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size
        )
    else:
        if rank == 0:
            logger.info("Using standard BYOL approach with augmentations...")
        train_dataset, val_dataset = FishDataset.get_datasets(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size
        )

    if rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")

    # Create distributed samplers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True, 
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False, 
            drop_last=False
        ) if val_dataset else None
        
        if rank == 0:
            logger.info(f"Training samples per GPU: {len(train_sampler)}")
            if val_sampler:
                logger.info(f"Validation samples per GPU: {len(val_sampler)}")
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=val_sampler,
            persistent_workers=True if args.num_workers > 0 else False
        )
    
    if rank == 0:
        logger.info(f"Training batches per GPU per epoch: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches per GPU per epoch: {len(val_loader)}")
        logger.info(f"Total training batches per epoch: {len(train_loader) * world_size}")

    return train_loader, val_loader, train_sampler

def main():
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()

    # Setup distributed training
    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    
    # Setup logger
    logger = setup_logger(rank)
    
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"Distributed training: {is_distributed}")
        logger.info(f"World size: {world_size}")
        if is_distributed:
            logger.info(f"Rank: {rank}, Local rank: {local_rank}")

    try:
        # Create data loaders
        train_loader, val_loader, train_sampler = create_data_loaders(
            args, is_distributed, rank, world_size
        )
        
        # Create model
        model = FishClassifier(
            model_path=args.model_path,
            backbone=args.backbone
        ).to(device)
        
        if rank == 0:
            logger.info(f"Using {args.backbone} as backbone network")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

        # Wrap model with DDP if distributed
        if is_distributed:
            model = DDP(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=False  # Set to True if you have unused parameters
            )
            if rank == 0:
                logger.info("Model wrapped with DistributedDataParallel")
        elif torch.cuda.device_count() > 1:
            logger.info(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel")
            model = nn.DataParallel(model)
        
        # Train model
        if rank == 0:
            logger.info(f"Starting training for {args.epochs} epochs...")
        
        _train_byol(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=args.save_path,
            use_class_pairs=args.use_class_pairs,
            device=device,
            is_distributed=is_distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            profile_batches=args.profile_batches,
            gradient_clip=args.gradient_clip,
            warmup_epochs=args.warmup_epochs
        )
        
        if rank == 0:
            logger.info(f"Training completed. Model saved to {args.save_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e
    finally:
        cleanup()

def _train_byol(model, train_loader, val_loader, train_sampler, epochs, lr, weight_decay, 
                save_path, use_class_pairs, device, is_distributed, rank, local_rank, 
                world_size, profile_batches=0, gradient_clip=1.0, warmup_epochs=5):
    """
    Enhanced training function using BYOL with comprehensive multi-GPU support and detailed time tracking
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        train_sampler: DistributedSampler for training data
        epochs: Number of training epochs
        lr: Base learning rate
        weight_decay: Weight decay for optimizer
        save_path: Path to save model
        use_class_pairs: Whether to use class-paired BYOL
        device: The device to use for training
        is_distributed: Boolean indicating if distributed training is used
        rank: Global rank of the current process
        local_rank: Local rank of the current process
        world_size: Total number of processes
        profile_batches: Number of batches to profile per epoch
        gradient_clip: Gradient clipping value
        warmup_epochs: Number of warmup epochs
    """
    logger = logging.getLogger(__name__)
    
    # Initialize enhanced time tracker
    time_tracker = TimeTracker(epochs, len(train_loader))
    
    if rank == 0:
        start_time_str = datetime.fromtimestamp(time_tracker.training_start_time).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Training started at: {start_time_str}")
        logger.info(f"Total epochs: {epochs}, Batches per epoch: {len(train_loader)}, Total steps: {time_tracker.total_steps}")
    
    # Get the actual model for optimization (unwrap DDP/DataParallel)
    model_to_optimize = model.module if isinstance(model, (DDP, nn.DataParallel)) else model

    # Scale learning rate by world size for distributed training
    scaled_lr = lr * world_size if is_distributed else lr
    
    # Setup optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': model_to_optimize.online_backbone.parameters(), 'lr': scaled_lr * 0.1},
        {'params': model_to_optimize.online_projector.parameters(), 'lr': scaled_lr},
        {'params': model_to_optimize.online_predictor.parameters(), 'lr': scaled_lr}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # Setup AMP for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 15  # Increased patience for multi-GPU training
    patience_counter = 0
    best_epoch = 0
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    
    if rank == 0:
        logger.info(f"Optimizer setup: Base LR={lr}, Scaled LR={scaled_lr}")
        logger.info(f"Warmup epochs: {warmup_epochs}, Total epochs: {epochs}")
        logger.info(f"Gradient clipping: {gradient_clip}")
        logger.info(f"Using mixed precision training: {scaler.is_enabled()}")
    
    # Training loop
    for epoch in range(epochs):
        # Start epoch tracking
        time_tracker.start_epoch(epoch)
        
        # Set epoch for distributed sampler to ensure different shuffling
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Setup profiler if needed
        prof = None
        if rank == 0 and profile_batches > 0 and epoch == 0:
            logger.info(f"Profiling first {profile_batches} batches of epoch 0...")
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=profile_batches, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_traces/train_byol'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.start()

        # Training loop
        for batch_idx, batch_data in enumerate(train_loader):
            # Start step tracking
            time_tracker.start_step()
            
            # Stop profiling after specified batches
            if rank == 0 and prof and batch_idx >= (1 + 1 + profile_batches):
                prof.stop()
                logger.info(f"Profiling finished. Trace saved to ./profiler_traces/train_byol")
                prof = None
            
            # Prepare batch data
            if use_class_pairs:
                anchor_images, positive_images, _ = batch_data
                anchor_images = anchor_images.to(device, non_blocking=True)
                positive_images = positive_images.to(device, non_blocking=True)
            else:
                src_images, _ = batch_data
                src_images = src_images.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                if use_class_pairs:
                    loss = model(anchor_images, positive_images)
                else:
                    loss = model(src_images)
                
                # Average loss if using DataParallel (not DDP)
                if not is_distributed and isinstance(model, nn.DataParallel):
                    loss = loss.mean()

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update target network (EMA update)
            if isinstance(model, (DDP, nn.DataParallel)):
                model.module.update_target_network()
            else:
                model.update_target_network()

            total_loss += loss.item()
            batch_count += 1
            
            # Complete step tracking
            time_tracker.complete_step()
            
            # Enhanced progress logging with time estimates
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                # Get step progress info
                step_info = time_tracker.get_step_progress_info()
                
                # Display progress every 10 steps or at important milestones
                should_log = (
                    batch_idx % 10 == 0 or  # Every 10 steps
                    batch_idx == len(train_loader) - 1 or  # Last batch
                    batch_idx < 5  # First few batches
                )
                
                if should_log and step_info:
                    # Format time information
                    current_step_time = TimeTracker.format_time(step_info['current_step_duration'])
                    avg_step_time = TimeTracker.format_time(step_info['avg_step_time'])
                    remaining_epoch_time = TimeTracker.format_time(step_info['remaining_time_in_epoch'])
                    remaining_total_time = TimeTracker.format_time(step_info['remaining_time_total'])
                    
                    # Basic progress info
                    logger.info(f"Epoch {step_info['epoch_progress']} | "
                               f"Step {step_info['step_progress']} | "
                               f"Loss: {loss.item():.4f} | "
                               f"LR: {current_lr:.2e}")
                    
                    # Time information
                    logger.info(f"Step: {current_step_time} | "
                               f"Avg: {avg_step_time} | "
                               f"Epoch ETA: {remaining_epoch_time} | "
                               f"Total ETA: {remaining_total_time} | "
                               f"Progress: {step_info['completion_percentage']:.1f}%")
                    
                # Quick progress for every step (less verbose)
                elif batch_idx % 50 == 0 and step_info:
                    remaining_total_time = TimeTracker.format_time(step_info['remaining_time_total'])
                    logger.info(f"E{epoch+1}/{epochs} B{batch_idx+1}/{len(train_loader)} | "
                               f"Loss: {loss.item():.4f} | "
                               f"ETA: {remaining_total_time} | "
                               f"{step_info['completion_percentage']:.1f}%")

            # Step profiler if active
            if rank == 0 and prof:
                prof.step()
        
        # Complete epoch tracking
        time_tracker.complete_epoch()
        
        # Cleanup profiler
        if rank == 0 and prof:
            prof.stop()
            logger.info(f"Profiling finished at end of epoch. Trace saved to ./profiler_traces/train_byol")
            prof = None

        # Calculate average training loss
        avg_train_loss = total_loss / batch_count
        
        # Synchronize training loss across all processes
        if is_distributed:
            avg_train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = avg_train_loss_tensor.item()

        train_losses.append(avg_train_loss)
        
        # Enhanced epoch summary with time estimates
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            epoch_info = time_tracker.get_epoch_progress_info()
            
            if epoch_info:
                # Format time information
                epoch_duration = TimeTracker.format_time(epoch_info['current_epoch_duration'])
                avg_epoch_time = TimeTracker.format_time(epoch_info['avg_epoch_time'])
                total_elapsed = TimeTracker.format_time(epoch_info['total_elapsed'])
                remaining_time = TimeTracker.format_time(epoch_info['estimated_remaining_time'])
                eta = TimeTracker.format_eta(epoch_info['eta'])
                
                logger.info("="*80)
                logger.info(f"EPOCH {epoch+1}/{epochs} COMPLETED")
                logger.info(f"📈 Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e}")
                logger.info(f"This Epoch: {epoch_duration} | Avg/Epoch: {avg_epoch_time}")
                logger.info(f"🕐 Elapsed: {total_elapsed} | Remaining: {remaining_time} | ETA: {eta}")
                logger.info(f"Progress: {epoch_info['completion_percentage']:.1f}% complete")
                logger.info("="*80)

        # Validation phase
        avg_val_loss = None
        if val_loader is not None:
            # Track validation time
            val_start_time = time.time()
            if rank == 0:
                logger.info(f"🔍 Starting validation for epoch {epoch+1}...")
            
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for val_batch_idx, batch_data in enumerate(val_loader):
                    if use_class_pairs:
                        anchor_images, positive_images, _ = batch_data
                        anchor_images = anchor_images.to(device, non_blocking=True)
                        positive_images = positive_images.to(device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda'):
                            loss = model(anchor_images, positive_images)
                    else:
                        src_images, _ = batch_data
                        src_images = src_images.to(device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda'):
                            loss = model(src_images)
                    
                    # Average loss if using DataParallel
                    if not is_distributed and isinstance(model, nn.DataParallel):
                        loss = loss.mean()

                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    # Progress update for validation
                    if rank == 0 and len(val_loader) > 20 and val_batch_idx % max(1, len(val_loader) // 5) == 0:
                        val_progress = (val_batch_idx + 1) / len(val_loader) * 100
                        val_elapsed = time.time() - val_start_time
                        val_eta = val_elapsed / (val_batch_idx + 1) * len(val_loader) - val_elapsed
                        logger.info(f"Validation progress: {val_batch_idx+1}/{len(val_loader)} "
                                   f"({val_progress:.1f}%) | ETA: {TimeTracker.format_time(val_eta)}")
            
            # Calculate validation duration
            val_duration = time.time() - val_start_time
            
            # Calculate average validation loss
            avg_val_loss = val_loss / val_batch_count
            
            # Synchronize validation loss across all processes
            if is_distributed:
                avg_val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)
                avg_val_loss = avg_val_loss_tensor.item()

            val_losses.append(avg_val_loss)

            # Early stopping and model saving (only on main process)
            if rank == 0:
                logger.info(f"✅ Validation completed in {TimeTracker.format_time(val_duration)}")
                logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    model_state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
                    
                    # Calculate current training duration
                    current_training_duration = time.time() - time_tracker.training_start_time
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'training_start_time': time_tracker.training_start_time,
                        'training_duration_seconds': current_training_duration,
                        'epoch_times': time_tracker.epoch_times,
                        'step_times': time_tracker.step_times,
                        'args': {
                            'backbone': model_to_optimize.backbone_name,
                            'lr': lr,
                            'world_size': world_size
                        }
                    }, save_path)
                    
                    logger.info(f"💾 Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"⚠️  Validation loss did not improve. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        logger.info(f"🛑 Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}")
                        break
        else:
            # No validation - save periodically
            if rank == 0 and ((epoch + 1) % 10 == 0 or epoch == epochs - 1):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                model_state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
                
                # Calculate current training duration
                current_training_duration = time.time() - time_tracker.training_start_time
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_losses': train_losses,
                    'training_start_time': time_tracker.training_start_time,
                    'training_duration_seconds': current_training_duration,
                    'epoch_times': time_tracker.epoch_times,
                    'step_times': time_tracker.step_times,
                    'args': {
                        'backbone': model_to_optimize.backbone_name,
                        'lr': lr,
                        'world_size': world_size
                    }
                }, save_path)
                
                logger.info(f"Model checkpoint saved at epoch {epoch+1}")
        
        # Step the scheduler
        scheduler.step()
        
        # Synchronize all processes before next epoch
        if is_distributed:
            dist.barrier()
    
    # Final model loading and saving
    if rank == 0:
        # Calculate final training duration
        final_training_duration = time.time() - time_tracker.training_start_time
        final_duration_str = str(timedelta(seconds=int(final_training_duration)))
        
        # Enhanced final training summary
        logger.info("="*100)
        logger.info("🏁 TRAINING COMPLETED!")
        logger.info("="*100)
        logger.info(f"Total Training Time: {final_duration_str}")
        
        if time_tracker.epoch_times:
            avg_epoch_time_final = sum(time_tracker.epoch_times) / len(time_tracker.epoch_times)
            avg_epoch_str = str(timedelta(seconds=int(avg_epoch_time_final)))
            logger.info(f"Average Time per Epoch: {avg_epoch_str}")
            
            # Additional statistics
            fastest_epoch = min(time_tracker.epoch_times)
            slowest_epoch = max(time_tracker.epoch_times)
            logger.info(f"⚡ Fastest Epoch: {TimeTracker.format_time(fastest_epoch)}")
            logger.info(f"🐌 Slowest Epoch: {TimeTracker.format_time(slowest_epoch)}")
        
        if time_tracker.step_times:
            avg_step_time = sum(time_tracker.step_times) / len(time_tracker.step_times)
            logger.info(f"🔄 Average Time per Step: {TimeTracker.format_time(avg_step_time)}")
            logger.info(f"📈 Total Steps Completed: {len(time_tracker.step_times)}")
            
            # Steps per second
            steps_per_second = len(time_tracker.step_times) / final_training_duration
            logger.info(f"🚀 Training Speed: {steps_per_second:.2f} steps/second")
        
        logger.info(f"📅 Training Started: {datetime.fromtimestamp(time_tracker.training_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🏁 Training Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if val_loader is not None and os.path.exists(save_path):
            # Load best model
            checkpoint = torch.load(save_path, map_location=device)
            current_model_to_load = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            current_model_to_load.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded best model from epoch {best_epoch+1}")
            logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")
            logger.info(f"📚 Training history: {len(train_losses)} epochs")
        elif val_loader is None:
            logger.info(f"✅ Training completed without validation")
        
        logger.info("="*100)

    # Final synchronization
    if is_distributed:
        dist.barrier()

    return save_path

if __name__ == "__main__":
    main() 