import os
import argparse
import setproctitle
import torch
import warnings
import datasets
import models
from task import train, test
from wsl import CMA
from utils import makedir, Logger, set_seed
import sys
warnings.filterwarnings("ignore")


def main(args):
    best_rank1 = 0
    best_mAP = 0
    best_epoch = 0
    patience_counter = 0
    
    # Patience settings
    phase1_patience = 12
    phase2_patience = 40
    max_patience = phase1_patience
    
    # Initialize logging (file-only, no terminal echo for verbose logs)
    log_path = os.path.join(args.save_path, "log/")
    model_path = os.path.join(args.save_path, "models/")
    makedir(log_path)
    makedir(model_path)
    
    logger = Logger(os.path.join(log_path, "log.txt"))
    
    if not args.resume and args.mode == 'train':
        logger.clear()
    
    # ==================== Header ====================
    header = '-' * 90 + '\n'
    header += f'WSL-VIReID Training Framework\n'
    header += f'Architecture: {args.arch} | Dataset: {args.dataset.upper()}\n'
    header += f'Mode: {args.mode.upper()}\n'
    header += '-' * 90
    
    print(header)  # Terminal
    logger(header)  # File
    # Initialize core components
    dataset = datasets.create(args)
    model = models.create(args)
    
    # ==================== Weight Loading ====================
    if args.load_weights:
        if os.path.isfile(args.load_weights):
            logger(f"=> Loading pretrained weights from '{args.load_weights}'")
            checkpoint = torch.load(args.load_weights, map_location=model.device)
            
            # Compatibility handling
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            if state_dict is not None:
                # Handle DataParallel 'module.' prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # strict=False: allows loading ResNet while ignoring Diffusion
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                logger(f"✓ Loaded successfully (Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)})")
            else:
                logger("❌ ERROR: Could not parse state_dict from checkpoint!")
        else:
            logger(f"⚠ WARNING: No checkpoint found at '{args.load_weights}'")
    
    # ==================== Training Mode ====================
    if args.mode == "train":
        cma = CMA(args)
        cma.to(model.device)
        
        if model.use_diffusion:
            model.diffusion.set_cma(cma)
            cma.set_ragm_module(model.ragm)
        
        start_epoch = 0
        
        # Resume from checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                logger(f"=> Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                best_rank1 = checkpoint['best_rank1']
                best_mAP = checkpoint['best_mAP']
                model.load_state_dict(checkpoint['state_dict'])
                
                if 'optimizer_phase1' in checkpoint and model.optimizer_phase1:
                    model.optimizer_phase1.load_state_dict(checkpoint['optimizer_phase1'])
                
                if 'optimizer_phase2' in checkpoint and hasattr(model, 'optimizer_phase2') and model.optimizer_phase2:
                    model.optimizer_phase2.load_state_dict(checkpoint['optimizer_phase2'])
                
                logger(f"✓ Loaded checkpoint (Epoch {checkpoint['epoch']})")
            else:
                logger(f"⚠ WARNING: No checkpoint found at '{args.resume}'")
        
        # Kickstart Phase 2 (if stage1_epoch == 0)
        if args.stage1_epoch == 0 and not hasattr(model, 'optimizer_phase2'):
            if hasattr(model, 'transition_to_phase2'):
                logger("=> [Kickstart] Stage 1 skipped. Initializing Phase 2 manually...")
                model.transition_to_phase2(args)
            else:
                logger("⚠ WARNING: 'transition_to_phase2' not found! Phase 2 might crash.")
        
        # ==================== Training Loop ====================
        # Calculate total iterations based on initial config
        total_epochs = args.stage1_epoch + args.stage2_epoch
        
        for epoch in range(start_epoch, total_epochs):
            # Dynamic Phase 1 check (allows stage1_epoch to be modified for early transition)
            enable_phase1 = (epoch < args.stage1_epoch)
            max_patience = phase1_patience if enable_phase1 else phase2_patience
            
            # Phase transition check
            # Logic: If we just finished stage 1 (epoch == args.stage1_epoch), switch.
            if epoch == args.stage1_epoch and args.stage1_epoch > 0:
                if hasattr(model, 'transition_to_phase2'):
                    logger(f"=> [Phase Transition] Epoch {epoch}: Switching to Phase 2...")
                    model.transition_to_phase2(args)
            
            # Train
            loss, loss_str = train(args, model, dataset, epoch, cma, logger, enable_phase1)
            
            # Validation Logic
            is_eval_epoch = ((epoch + 1) % args.eval_step == 0) or ((epoch + 1) == total_epochs) or ((epoch + 1) == args.stage1_epoch)
            
            if is_eval_epoch:
                print(f"\n🔍 Epoch {epoch+1} Evaluation...")  # Terminal only
                cmc, mAP, mINP = test(args, model, dataset, logger)
                
                # Convert to percentage
                rank1 = cmc[0] * 100
                mAP = mAP * 100
                mINP = mINP * 100
                
                is_best = (rank1 > best_rank1)
                
                if is_best:
                    best_rank1 = rank1
                    best_mAP = mAP
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    result_msg = f"✓ New Best! Epoch {epoch+1} | R1: {rank1:.2f}% | mAP: {mAP:.2f}% | mINP: {mINP:.2f}%"
                    print(result_msg)  # Terminal
                    logger(result_msg)  # File
                    
                    save_state = {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_rank1': best_rank1,
                        'best_mAP': best_mAP,
                    }

                    if hasattr(model, 'optimizer_phase1') and model.optimizer_phase1 is not None:
                        save_state['optimizer_phase1'] = model.optimizer_phase1.state_dict()
                    if hasattr(model, 'optimizer_phase2') and model.optimizer_phase2 is not None:
                        save_state['optimizer_phase2'] = model.optimizer_phase2.state_dict()

                    torch.save(save_state, os.path.join(model_path, 'checkpoint.pth'))
                    torch.save(save_state, os.path.join(model_path, 'checkpoint_best.pth'))

                else:
                    patience_counter += 1
                    result_msg = f"Epoch {epoch+1} | R1: {rank1:.2f}% | mAP: {mAP:.2f}% | Best: {best_rank1:.2f}% (Ep {best_epoch}) | Patience: {patience_counter}/{max_patience}"
                    print(result_msg)  # Terminal
                    logger(result_msg)  # File
                
                logger('-' * 80)
                
                # ==================== Early Stopping Logic (FIXED) ====================
                if patience_counter >= max_patience:
                    if enable_phase1:
                        # 🔴 FIX: Phase 1 Early Stopping -> Transition to Phase 2
                        logger(f"⚠️ Phase 1 Early Stopping triggered at Epoch {epoch+1}.")
                        logger(f"=> Loading best weights from Epoch {best_epoch} and transitioning to Phase 2 immediately.")
                        
                        # 1. Load best weights
                        best_ckpt_path = os.path.join(model_path, 'checkpoint_best.pth')
                        if os.path.isfile(best_ckpt_path):
                            checkpoint = torch.load(best_ckpt_path)
                            model.load_state_dict(checkpoint['state_dict'])
                        
                        # 2. Reset patience
                        patience_counter = 0
                        best_rank1 = 0 # Reset best metrics for Phase 2 tracking
                        
                        # 3. Force Transition in next iteration
                        # By setting stage1_epoch to current epoch + 1, 
                        # the next loop iteration (epoch + 1) will satisfy:
                        #   enable_phase1 = (epoch+1 < epoch+1) -> False
                        #   epoch+1 == args.stage1_epoch -> True (Trigger Transition)
                        args.stage1_epoch = epoch + 1
                        
                        # Do NOT break
                    else:
                        # Phase 2 Early Stopping -> Real Stop
                        stop_msg = f"⏹ Early stopping triggered at Epoch {epoch+1} (Phase 2)."
                        print(stop_msg)
                        logger(stop_msg)
                        break
    
    # ==================== Testing Mode ====================
    elif args.mode == "test":
        logger("Starting testing...")
        
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            logger(f"✓ Loaded weights from '{args.resume}'")
        
        cmc, mAP, mINP = test(args, model, dataset, logger)
        
        result = f"Test Result | Rank-1: {cmc[0]*100:.2f}% | mAP: {mAP*100:.2f}% | mINP: {mINP*100:.2f}%"
        print(result)
        logger(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSL-VIReID Training')
    
    # Dataset & Basic
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu or llcm')
    parser.add_argument('--lr', default=0.00035, type=float, help='learning rate')
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70])
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup-epoch', default=10, type=int)
    
    # Model
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--load-weights', type=str, default='')
    parser.add_argument('--batch-pidnum', default=12, type=int)
    parser.add_argument('--pid-numsample', default=4, type=int)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=1, type=int)
    
    # Loss weights
    parser.add_argument('--tri-weight', default=1.0, type=float)
    parser.add_argument('--weak-weight', default=0.5, type=float)
    parser.add_argument('--ccpa-weight', default=1.0, type=float)
    parser.add_argument('--diffusion-weight', default=0.05, type=float)
    
    # Epoch settings
    parser.add_argument('--stage1-epoch', default=50, type=int)
    parser.add_argument('--stage2-epoch', default=120, type=int)
    parser.add_argument('--ccpa-start-epoch', default=60, type=int)
    
    # CMA settings
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--ccpa-threshold-mode', default='hybrid', type=str)
    parser.add_argument('--pseudo-momentum', default=0.9, type=float)
    parser.add_argument('-T', '--temperature', default=0.07, type=float)
    
    # Diffusion settings
    parser.add_argument('--use-diffusion', action='store_true')
    parser.add_argument('--feature-diffusion-steps', default=5, type=int)
    parser.add_argument('--semantic-diffusion-steps', default=10, type=int)
    parser.add_argument('--diffusion-hidden', default=1024, type=int)
    parser.add_argument('--diffusion-lr', default=0.001, type=float)
    
    # Reliability Gating settings
    parser.add_argument('--use-memory-bank', action='store_true')
    parser.add_argument('--memory-size-per-class', default=5, type=int)
    parser.add_argument('--ragm-temperature', default=0.1, type=float)
    
    # Path settings
    parser.add_argument('--save-path', default='save', type=str)
    parser.add_argument('--data-path', default='/root/autodl-tmp/data/', type=str)
    
    # Other settings
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--eval-step', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--use-cosine-annealing', action='store_true')
    parser.add_argument('--use-cycle-consistency', action='store_true')
    parser.add_argument('--test-batch', default=64, type=int)
    parser.add_argument('--debug', default='wsl', type=str)
    parser.add_argument('--model-path', default='default', type=str)
    
    # Dataset specific
    parser.add_argument('--trial', default=1, type=int)
    parser.add_argument('--search-mode', default='all', type=str)
    parser.add_argument('--gall-mode', default='single', type=str)
    parser.add_argument('--test-mode', default='t2v', type=str)
    parser.add_argument('--img-h', default=288, type=int)
    parser.add_argument('--img-w', default=144, type=int)
    parser.add_argument('--relabel', default=1, type=int)
    
    # Cross attention settings
    parser.add_argument('--cross-attn-heads', default=4, type=int)
    parser.add_argument('--cross-attn-dropout', default=0.1, type=float)
    
    args = parser.parse_args()
    args.num_workers = args.workers
    
    # Path setup
    args.save_path = './saved_' + args.dataset + '_{}'.format(args.arch) + '/' + args.save_path
    
    # Dataset-specific classes
    if args.dataset == 'sysu':
        args.num_classes = 395
    elif args.dataset == 'regdb':
        args.num_classes = 206
    elif args.dataset == 'llcm':
        args.num_classes = 713
    
    set_seed(args.seed)
    setproctitle.setproctitle(f'wsl-vireid_{args.dataset}')
    
    main(args)