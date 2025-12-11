"""
Cross-Modal Person Re-Identification Main Entry
================================================
Author: Fixed Version
Date: 2025-12-11
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
    best_rank1 = 0
    best_mAP = 0
    
    log_path = os.path.join(args.save_path, "log/")
    model_path = os.path.join(args.save_path, "models/")
    makedir(log_path)
    makedir(model_path)
    
    logger = Logger(os.path.join(log_path, "log.txt"))
    if not args.resume and args.mode == 'train':
        logger.clear()
    logger(args)
    
    dataset = datasets.create(args)
    model = models.create(args)

    if args.mode == "train":
        cma = CMA(args)
        
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

        if enable_phase1:
            logger('Time: {} | Starting Phase 1 from epoch 0'.format(time_now()))
            for current_epoch in range(0, args.stage1_epoch):
                model.scheduler_phase1.step(current_epoch)
                _, result = train(args, model, dataset, current_epoch, cma, logger, enable_phase1)
                cmc, mAP, mINP = test(args, model, dataset, current_epoch)
                
                best_rank1 = max(cmc[0], best_rank1)
                best_mAP = max(mAP, best_mAP)
                
                logger('Time: {} | Phase 1 Epoch {}; Setting: {}'.format(
                    time_now(), current_epoch + 1, args.save_path))
                logger(f'Learning Rate: {model.scheduler_phase1.get_lr()[0]}')
                logger(result)
                logger('R1:{:.4f}; R10:{:.4f}; R20:{:.4f}; mAP:{:.4f}; mINP:{:.4f}\n'
                       'Best R1: {:.4f}; Best mAP: {:.4f}'.format(
                           cmc[0], cmc[9], cmc[19], mAP, mINP, best_rank1, best_mAP))
                logger('=' * 50)
                
                if current_epoch == args.stage1_epoch - 1:
                    save_checkpoint(args, model, current_epoch + 1)
        
        enable_phase1 = False
        start_epoch = model.resume_epoch
        logger('Time: {} | Starting Phase 2 from epoch {}'.format(time_now(), start_epoch))
        
        for current_epoch in range(start_epoch, args.stage2_epoch):
            model.scheduler_phase2.step(current_epoch)
            _, result = train(args, model, dataset, current_epoch, cma, logger, enable_phase1)
            cmc, mAP, mINP = test(args, model, dataset, current_epoch)
            
            is_best_rank = (cmc[0] >= best_rank1)
            best_rank1 = max(cmc[0], best_rank1)
            best_mAP = max(mAP, best_mAP)
            
            model.save_model(current_epoch, is_best_rank)
            
            logger('=' * 50)
            logger('Epoch: {}; Time: {}; Setting: {}'.format(
                current_epoch, time_now(), args.save_path))
            logger(f'Learning Rate: {model.scheduler_phase2.get_lr()[0]}')
            logger(result)
            logger('R1:{:.4f}; R10:{:.4f}; R20:{:.4f}; mAP:{:.4f}; mINP:{:.4f}\n'
                   'Best R1: {:.4f}; Best mAP: {:.4f}'.format(
                       cmc[0], cmc[9], cmc[19], mAP, mINP, best_rank1, best_mAP))
            logger('=' * 50)

    if args.mode == 'test':
        if args.model_path == 'default':
            model.resume_model()
        else:
            model.resume_model(args.model_path)
        
        cmc, mAP, mINP = test(args, model, dataset)
        logger('Time: {}; Testing on Dataset: {}'.format(time_now(), args.dataset))
        logger('R1:{:.4f}; R10:{:.4f}; R20:{:.4f}; mAP:{:.4f}; mINP:{:.4f}'.format(
            cmc[0], cmc[9], cmc[19], mAP, mINP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSL-ReID")
    parser.add_argument("--dataset", default="regdb", type=str,
                        help="Dataset name: sysu, llcm, regdb")
    parser.add_argument("--arch", default="resnet", type=str,
                        help="Backbone architecture")
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument("--data-path", default="./datasets", type=str,
                        help="Dataset root path")
    parser.add_argument("--save-path", default="save/", type=str,
                        help="Log and model save path")
    
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help='Weight decay')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 70],
                        help='Learning rate decay milestones')
    parser.add_argument('--relabel', default=1, type=int, help='Relabel training dataset')
    parser.add_argument('--weak-weight', default=0.25, type=float,
                        help='Weight of weak loss')
    parser.add_argument('--tri-weight', default=0.25, type=float,
                        help='Weight of triplet loss')
    
    parser.add_argument('--img-h', default=288, type=int, help='Input image height')
    parser.add_argument('--img-w', default=144, type=int, help='Input image width')
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers')
    parser.add_argument('--batch-pidnum', default=8, type=int,
                        help='Number of person IDs per batch')
    parser.add_argument('--pid-numsample', default=4, type=int,
                        help='Number of samples per person ID')
    parser.add_argument('--test-batch', default=128, type=int, metavar='tb',
                        help='Testing batch size')
    
    parser.add_argument('--sigma', default=0.8, type=float,
                        help='Momentum update factor for memory bank')
    parser.add_argument('-T', '--temperature', default=3, type=float,
                        help='Temperature parameter for softmax')
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument('--stage1-epoch', default=20, type=int,
                        help='Number of epochs for Phase 1')
    parser.add_argument('--stage2-epoch', default=120, type=int,
                        help='Number of epochs for Phase 2')
    parser.add_argument('--resume', default=0, type=int, help='Resume training')
    parser.add_argument('--debug', default='wsl', type=str, help='Debug mode: wsl or sl')
    parser.add_argument('--trial', default=1, type=int, help='Trial number for RegDB')
    parser.add_argument('--search-mode', default='all', type=str,
                        help='Search gallery mode')
    parser.add_argument('--gall-mode', default='single', type=str,
                        help='Gallery mode: single or multi shot')
    parser.add_argument('--test-mode', default='t2v', type=str,
                        help='Test mode for RegDB and LLCM')
    parser.add_argument('--model-path', default='default', type=str,
                        help='Path to load checkpoint')
    
    parser.add_argument('--use-clip', action='store_true',
                        help='Enable CLIP as semantic referee')
    parser.add_argument('--use-sinkhorn', action='store_true',
                        help='Use Sinkhorn algorithm for global matching')
    parser.add_argument('--sinkhorn-reg', type=float, default=0.05,
                        help='Entropy regularization for Sinkhorn')
    parser.add_argument('--w-clip', type=float, default=0.3,
                        help='Weight for CLIP similarity in matching')
    
    args = parser.parse_args()
    
    args.save_path = './saved_' + args.dataset + '_{}'.format(args.arch) + '/' + args.save_path
    if args.dataset == 'sysu':
        args.num_classes = 395
    elif args.dataset == 'regdb':
        args.num_classes = 206
        args.save_path += f'_{args.trial}'
    elif args.dataset == 'llcm':
        args.num_classes = 713
    
    set_seed(args.seed)
    setproctitle.setproctitle(args.save_path)
    main(args)
