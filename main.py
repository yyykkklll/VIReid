"""
Cross-Modal Person Re-Identification Main Entry
================================================
Optimizations:
1. Skip early epoch testing to avoid "No valid query" error
2. Adaptive testing frequency (Phase 1: every 5 epochs, Phase 2: every 3 epochs)
3. Better exception handling for testing failures
4. Improved logging with emojis and formatting
5. Automatic device detection (no manual GPU selection needed)
6. Fixed learning rate scheduler step timing
7. Better checkpoint management
8. BUG FIX: Force enable gradient if testing fails
"""

import os
import argparse
import setproctitle
import torch
import warnings
import datasets
import models
from task import train, test
from wsl import CMA
from utils import time_now, makedir, Logger, set_seed, save_checkpoint

warnings.filterwarnings("ignore")


def main(args):
    """Main training and testing workflow"""
    
    # Setup paths
    log_path = os.path.join(args.save_path, "log/")
    model_path = os.path.join(args.save_path, "models/")
    makedir(log_path)
    makedir(model_path)
    
    # Setup logger
    logger = Logger(os.path.join(log_path, "log.txt"))
    if not args.resume and args.mode == 'train':
        logger.clear()
    
    logger("="*80)
    logger("WSL-ReID Training Configuration")
    logger("="*80)
    logger(args)
    logger("="*80)
    
    # Create dataset and model
    dataset = datasets.create(args)
    model = models.create(args)
    
    if args.mode == "train":
        cma = CMA(args)
        
        # Determine starting point
        if args.resume or not args.model_path == 'default':
            enable_phase1 = False
            if 'wsl' in args.debug and not args.model_path == 'default':
                model.resume_model(args.model_path)
            else:
                model.resume_model()
        elif 'wsl' in args.debug:
            enable_phase1 = True
            model.resume_model()
        else:
            enable_phase1 = False
            model.resume_model()
        
        # Initialize best metrics
        best_rank1 = 0.0
        best_mAP = 0.0
        best_epoch = 0
        
        # ==================== Phase 1: Intra-modal Learning ====================
        if enable_phase1:
            logger("\n" + "="*80)
            logger("🚀 PHASE 1: Intra-Modal Learning")
            logger("="*80)
            logger(f"Time: {time_now()} | Starting from epoch 0")
            logger(f"Total epochs: {args.stage1_epoch}")
            logger("="*80 + "\n")
            
            for current_epoch in range(0, args.stage1_epoch):
                logger(f"\n{'='*80}")
                logger(f"Phase 1 - Epoch {current_epoch + 1}/{args.stage1_epoch}")
                logger(f"{'='*80}")
                
                # Training
                _, result = train(args, model, dataset, current_epoch + 1, cma, logger, enable_phase1=True)
                
                # Step scheduler after first epoch
                if current_epoch > 0:
                    model.scheduler_phase1.step()
                
                current_lr = model.scheduler_phase1.get_last_lr()[0]
                logger(f"Learning Rate: {current_lr:.6f}")
                logger(f"Training Losses:\n{result}")
                
                # Testing strategy: skip first 5 epochs, then test every 5 epochs
                if (current_epoch + 1) < 5:
                    logger(f"⏩ Skip testing (warmup period: {current_epoch + 1}/5)")
                    continue
                
                if (current_epoch + 1) % 5 != 0 and current_epoch != args.stage1_epoch - 1:
                    logger(f"⏩ Skip testing (test every 5 epochs)")
                    continue
                
                # Perform testing
                logger(f"\n{'='*80}")
                logger(f"📊 Testing at Epoch {current_epoch + 1}...")
                logger(f"{'='*80}")
                
                try:
                    cmc, mAP, mINP = test(args, model, dataset, current_epoch + 1)
                    
                    # Update best metrics
                    is_best = (cmc[0] > best_rank1)
                    if is_best:
                        best_rank1 = cmc[0]
                        best_mAP = mAP
                        best_epoch = current_epoch + 1
                    
                    # Log results
                    logger(f"\n📈 Epoch {current_epoch + 1} Results:")
                    logger(f"  Rank-1 : {cmc[0]:.2%}")
                    logger(f"  Rank-5 : {cmc[4]:.2%}")
                    logger(f"  Rank-10: {cmc[9]:.2%}")
                    logger(f"  Rank-20: {cmc[19]:.2%}")
                    logger(f"  mAP    : {mAP:.2%}")
                    logger(f"  mINP   : {mINP:.2%}")
                    logger(f"\n🏆 Best Rank-1: {best_rank1:.2%} at epoch {best_epoch}")
                    
                    # Save checkpoint
                    if is_best:
                        save_checkpoint(args, model, current_epoch + 1)
                        logger(f"✅ Best model saved!")
                    
                    # Periodic save
                    if (current_epoch + 1) % 10 == 0:
                        save_checkpoint(args, model, current_epoch + 1)
                        logger(f"💾 Checkpoint saved at epoch {current_epoch + 1}")
                
                except Exception as e:
                    logger(f"\n⚠️  Testing failed: {str(e)}")
                    # FIX: 强制恢复梯度计算，防止后续训练 epoch 崩溃
                    torch.set_grad_enabled(True)
                    logger("Continuing training...")
                
                logger("="*80)
            
            # Save Phase 1 final model
            save_checkpoint(args, model, args.stage1_epoch)
            logger(f"\n✅ Phase 1 completed! Best Rank-1: {best_rank1:.2%}\n")
        
        # ==================== Phase 2: Cross-Modal Learning ====================
        enable_phase1 = False
        start_epoch = model.resume_epoch if hasattr(model, 'resume_epoch') else 0
        
        logger("\n" + "="*80)
        logger("🚀 PHASE 2: Cross-Modal Learning")
        logger("="*80)
        logger(f"Time: {time_now()} | Starting from epoch {start_epoch}")
        logger(f"Total epochs: {args.stage2_epoch}")
        logger(f"Current Best Rank-1: {best_rank1:.2%}")
        logger("="*80 + "\n")
        
        for current_epoch in range(start_epoch, args.stage2_epoch):
            logger(f"\n{'='*80}")
            logger(f"Phase 2 - Epoch {current_epoch + 1}/{args.stage2_epoch}")
            logger(f"{'='*80}")
            
            # Training
            _, result = train(args, model, dataset, current_epoch + 1, cma, logger, enable_phase1=False)
            
            # Step scheduler after first epoch
            if current_epoch > start_epoch:
                model.scheduler_phase2.step()
            
            current_lr = model.scheduler_phase2.get_last_lr()[0]
            logger(f"Learning Rate: {current_lr:.6f}")
            logger(f"Training Losses:\n{result}")
            
            # Testing strategy: skip first 3 epochs, then test every 3 epochs
            if (current_epoch + 1) < 3:
                logger(f"⏩ Skip testing (warmup period: {current_epoch + 1}/3)")
                continue
            
            if (current_epoch + 1) % 3 != 0 and current_epoch != args.stage2_epoch - 1:
                logger(f"⏩ Skip testing (test every 3 epochs)")
                continue
            
            # Perform testing
            logger(f"\n{'='*80}")
            logger(f"📊 Testing at Epoch {current_epoch + 1}...")
            logger(f"{'='*80}")
            
            try:
                cmc, mAP, mINP = test(args, model, dataset, current_epoch + 1)
                
                # Update best metrics
                is_best = (cmc[0] > best_rank1)
                if is_best:
                    best_rank1 = cmc[0]
                    best_mAP = mAP
                    best_epoch = current_epoch + 1
                
                # Log results
                logger(f"\n📈 Epoch {current_epoch + 1} Results:")
                logger(f"  Rank-1 : {cmc[0]:.2%}")
                logger(f"  Rank-5 : {cmc[4]:.2%}")
                logger(f"  Rank-10: {cmc[9]:.2%}")
                logger(f"  Rank-20: {cmc[19]:.2%}")
                logger(f"  mAP    : {mAP:.2%}")
                logger(f"  mINP   : {mINP:.2%}")
                logger(f"\n🏆 Best Rank-1: {best_rank1:.2%} at epoch {best_epoch}")
                
                # Save checkpoint
                model.save_model(current_epoch + 1, is_best_rank=is_best)
                if is_best:
                    logger(f"✅ Best model saved!")
                
                # Periodic save
                if (current_epoch + 1) % 10 == 0:
                    logger(f"💾 Checkpoint saved at epoch {current_epoch + 1}")
            
            except Exception as e:
                logger(f"\n⚠️  Testing failed: {str(e)}")
                # FIX: 强制恢复梯度计算
                torch.set_grad_enabled(True)
                logger("Continuing training...")
            
            logger("="*80)
        
        # Final summary
        logger("\n" + "="*80)
        logger("🎉 TRAINING COMPLETED!")
        logger("="*80)
        logger(f"Final Best Rank-1: {best_rank1:.2%} at epoch {best_epoch}")
        logger(f"Final Best mAP: {best_mAP:.2%}")
        logger(f"Time: {time_now()}")
        logger("="*80 + "\n")
    
    # ==================== Testing Mode ====================
    if args.mode == 'test':
        logger("\n" + "="*80)
        logger("🔍 TESTING MODE")
        logger("="*80)
        
        if args.model_path == 'default':
            model.resume_model()
            logger("Loading default checkpoint...")
        else:
            model.resume_model(args.model_path)
            logger(f"Loading checkpoint from: {args.model_path}")
        
        logger(f"Dataset: {args.dataset}")
        logger(f"Test mode: {args.test_mode}")
        logger("="*80)
        
        cmc, mAP, mINP = test(args, model, dataset)
        
        logger(f"\n📊 Test Results:")
        logger(f"  Rank-1 : {cmc[0]:.2%}")
        logger(f"  Rank-5 : {cmc[4]:.2%}")
        logger(f"  Rank-10: {cmc[9]:.2%}")
        logger(f"  Rank-20: {cmc[19]:.2%}")
        logger(f"  mAP    : {mAP:.2%}")
        logger(f"  mINP   : {mINP:.2%}")
        logger(f"\nTime: {time_now()}")
        logger("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "WSL-ReID",
        description="Cross-Modal Person Re-Identification with Weak Supervision Learning"
    )
    
    # ==================== Basic Settings ====================
    parser.add_argument("--dataset", default="regdb", type=str,
                        choices=["sysu", "llcm", "regdb"],
                        help="Dataset name")
    parser.add_argument("--arch", default="resnet", type=str,
                        choices=["resnet", "clip"],
                        help="Backbone architecture")
    parser.add_argument('--mode', default='train', type=str,
                        choices=['train', 'test'],
                        help='Training or testing mode')
    parser.add_argument("--data-path", default="./datasets", type=str,
                        help="Dataset root path")
    parser.add_argument("--save-path", default="./save", type=str,
                        help="Log and model checkpoint save path")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of data loading workers')
    
    # ==================== Training Phases ====================
    parser.add_argument('--stage1-epoch', default=40, type=int,
                        help='Number of epochs for Phase 1 (intra-modal learning)')
    parser.add_argument('--stage2-epoch', default=60, type=int,
                        help='Number of epochs for Phase 2 (cross-modal learning)')
    parser.add_argument('--resume', default=0, type=int,
                        help='Resume from checkpoint (1: yes, 0: no)')
    parser.add_argument('--model-path', default='default', type=str,
                        help='Path to load checkpoint for testing or resuming')
    
    # ==================== Data Settings ====================
    parser.add_argument('--img-h', default=288, type=int,
                        help='Input image height')
    parser.add_argument('--img-w', default=144, type=int,
                        help='Input image width')
    parser.add_argument('--batch-pidnum', default=16, type=int,
                        help='Number of person IDs per batch')
    parser.add_argument('--pid-numsample', default=4, type=int,
                        help='Number of samples per person ID')
    parser.add_argument('--test-batch', default=128, type=int,
                        help='Testing batch size')
    parser.add_argument('--relabel', default=1, type=int,
                        help='Relabel training dataset (1: yes, 0: no)')
    
    # ==================== Optimizer Settings ====================
    parser.add_argument('--lr', default=0.0003, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help='Weight decay for optimizer')
    
    # ==================== Learning Rate Scheduler ====================
    parser.add_argument('--use-cosine-scheduler', action='store_true',
                        help='Use cosine annealing scheduler instead of multistep')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 50],
                        help='Milestones for multistep scheduler')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--eta-min', default=1e-6, type=float,
                        help='Minimum learning rate for cosine scheduler')
    
    # ==================== Loss Function Settings ====================
    parser.add_argument('--tri-weight', default=0.25, type=float,
                        help='Weight of triplet loss')
    parser.add_argument('--weak-weight', default=0.25, type=float,
                        help='Weight of weak supervision loss')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='Label smoothing epsilon (0 to disable)')
    parser.add_argument('--contrastive-temp', default=0.07, type=float,
                        help='Temperature for InfoNCE contrastive loss')
    parser.add_argument('--intra-contrastive-weight', default=0.1, type=float,
                        help='Weight for intra-modal contrastive loss')
    parser.add_argument('--contrastive-weight', default=0.3, type=float,
                        help='Weight for cross-modal contrastive loss')
    parser.add_argument('--contrastive-start-epoch', default=5, type=int,
                        help='Epoch to start intra-modal contrastive loss')
    parser.add_argument('--cross-contrastive-start', default=3, type=int,
                        help='Epoch to start cross-modal contrastive loss')
    parser.add_argument('--triplet-warmup-epochs', default=8, type=int,
                        help='Warmup epochs for triplet loss')
    parser.add_argument('--cmo-start-epoch', default=3, type=int,
                        help='Epoch to start CMO loss in phase 2')
    parser.add_argument('--cmo-warmup', default=8, type=int,
                        help='Warmup epochs for CMO loss')
    parser.add_argument('--weak-start-epoch', default=5, type=int,
                        help='Epoch to start weak supervision loss')
    parser.add_argument('--weak-warmup', default=12, type=int,
                        help='Warmup epochs for weak supervision loss')
    
    # ==================== Cross-Modal Matching ====================
    parser.add_argument('--sigma', default=0.3, type=float,
                        help='Momentum update factor for memory bank')
    parser.add_argument('-T', '--temperature', default=0.05, type=float,
                        help='Temperature parameter for softmax')
    
    # ==================== Advanced Features ====================
    parser.add_argument('--use-clip', action='store_true',
                        help='Enable CLIP as semantic referee')
    parser.add_argument('--w-clip', type=float, default=0.3,
                        help='Weight for CLIP similarity in matching')
    parser.add_argument('--use-sinkhorn', action='store_true',
                        help='Use Sinkhorn algorithm for global optimal matching')
    parser.add_argument('--sinkhorn-reg', type=float, default=0.05,
                        help='Entropy regularization for Sinkhorn algorithm')
    
    # ==================== Testing Settings ====================
    parser.add_argument('--test-mode', default='all', type=str,
                        help='Test mode: all, t2v, v2t')
    parser.add_argument('--search-mode', default='all', type=str,
                        help='Search gallery mode')
    parser.add_argument('--gall-mode', default='single', type=str,
                        choices=['single', 'multi'],
                        help='Gallery mode: single or multi shot')
    parser.add_argument('--trial', default=1, type=int,
                        help='Trial number for RegDB dataset')
    
    # ==================== Debug Settings ====================
    parser.add_argument('--debug', default='wsl', type=str,
                        choices=['wsl', 'sl'],
                        help='Debug mode: wsl (weak supervision) or sl (strong supervision)')
    
    args = parser.parse_args()
    
    # ==================== Setup ====================
    # Configure save path
    args.save_path = './saved_' + args.dataset + '_{}'.format(args.arch) + '/' + args.save_path
    
    # Dataset-specific settings
    if args.dataset == 'sysu':
        args.num_classes = 395
    elif args.dataset == 'regdb':
        args.num_classes = 206
        args.save_path += f'_{args.trial}'
    elif args.dataset == 'llcm':
        args.num_classes = 713
    
    # Set random seed
    set_seed(args.seed)
    
    # Set process title
    setproctitle.setproctitle(f"WSL-ReID-{args.dataset}")
    
    # Run main
    main(args)