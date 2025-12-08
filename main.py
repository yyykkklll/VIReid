import os
import argparse
import setproctitle
import torch
import warnings
import time
import copy # [新增]

import datasets
import models
from task import train, test
from models.sg_module import SGM 
from utils import time_now, makedir, Logger, set_seed, save_checkpoint

warnings.filterwarnings("ignore")

def main(args):
    best_rank1 = 0
    best_mAP = 0
    
    log_path = os.path.join(args.save_path, "log/")
    model_path = os.path.join(args.save_path, "models/")
    makedir(log_path)
    makedir(model_path)
    
    logger = Logger(os.path.join(log_path, "log.txt"))
    if not args.resume and args.mode == 'train' and args.model_path == 'default':
        logger.clear()
        
    logger(args)
    
    dataset = datasets.create(args)
    model = models.create(args)

    if args.mode == "train":
        sgm = SGM(args).to(model.device)
        
        enable_phase1 = False
        if args.resume or args.model_path != 'default':
            model.resume_model(args.model_path)
            if 'wsl' in args.debug and args.model_path == 'default' and model.resume_epoch == 0:
                 enable_phase1 = True
        elif 'wsl' in args.debug:
            enable_phase1 = True
        
        # ======================================================
        # Phase 1: 单模态预热
        # ======================================================
        if enable_phase1:
            logger('Time: {} | [Phase 1] Intra-modal Warmup Start'.format(time_now()))
            for current_epoch in range(0, args.stage1_epoch):
                model.scheduler_phase1.step(current_epoch)
                
                _, result = train(args, model, dataset, current_epoch, sgm, logger, enable_phase1=True)
                
                cmc, mAP, mINP = test(args, model, dataset, current_epoch) 
                best_rank1 = max(cmc[0], best_rank1)
                best_mAP = max(mAP, best_mAP)
                
                logger('Time: {} | Phase 1 Epoch {}; Setting: {}'.format(time_now(), current_epoch+1, args.save_path))
                logger(f'LR: {model.scheduler_phase1.get_lr()[0]}')
                logger(result)
                logger('R1:{:.2f} | R10:{:.2f} | mAP:{:.2f} | Best R1:{:.2f}'.format(
                    cmc[0]*100, cmc[9]*100, mAP*100, best_rank1*100))
                logger('=' * 50)
                
                if current_epoch == args.stage1_epoch - 1:
                    save_checkpoint(args, model, current_epoch + 1)

        # ======================================================
        # Phase 2: 结构感知对齐
        # ======================================================
        enable_phase1 = False
        start_epoch = model.resume_epoch
        if start_epoch < args.stage1_epoch and 'wsl' in args.debug and not args.resume:
             start_epoch = args.stage1_epoch

        logger('Time: {} | [Phase 2] Structure-Aware Alignment Start from Epoch {}'.format(time_now(), start_epoch))
        
        # [核心创新点实现]：专家权重继承 (Expert Inheritance)
        # 在进入 Phase 2 之前，将 Classifier1 (RGB Expert) 的权重复制给 Classifier3 (Shared)
        # 避免 Classifier3 冷启动导致的崩塌
        if model.classifier3 is not None and model.classifier1 is not None:
            logger('>>> [Init] Initializing Shared Classifier from RGB Expert...')
            model.classifier3.load_state_dict(model.classifier1.state_dict())
            
            # 由于优化器已经初始化过了，我们需要确保优化器状态也是健康的
            # 这里简单起见，我们让 classifier3 继承权重，然后在后续训练中微调
        
        for current_epoch in range(start_epoch, args.stage2_epoch):
            model.scheduler_phase2.step(current_epoch)
            
            _, result = train(args, model, dataset, current_epoch, sgm, logger, enable_phase1=False)

            cmc, mAP, mINP = test(args, model, dataset, current_epoch) 
            
            is_best_rank = (cmc[0] >= best_rank1)
            best_rank1 = max(cmc[0], best_rank1)
            best_mAP = max(mAP, best_mAP)
            
            model.save_model(current_epoch + 1, is_best_rank)
            
            logger('=' * 50)
            logger('Epoch: {} | Time: {}'.format(current_epoch + 1, time_now()))
            logger(f'LR: {model.scheduler_phase2.get_lr()[0]}')
            logger(result)
            logger('R1:{:.2f} | R10:{:.2f} | mAP:{:.2f} | Best R1:{:.2f}'.format(
                    cmc[0]*100, cmc[9]*100, mAP*100, best_rank1*100))
            logger('=' * 50)
        
    if args.mode == 'test':
        model.resume_model(args.model_path)
        cmc, mAP, mINP = test(args, model, dataset)
        logger('Time: {}; Test on Dataset: {}'.format(time_now(), args.dataset))
        logger('R1:{:.2f} | R10:{:.2f} | mAP:{:.2f}'.format(
               cmc[0]*100, cmc[9]*100, mAP*100))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SG-WSL: Semantic-Graph Weakly Supervised ReID")
    parser.add_argument("--dataset", default="regdb", type=str, help="dataset name")
    parser.add_argument("--data-path", default="/root/vireid/datasets", type=str)
    parser.add_argument("--arch", default="vit", type=str, help="backbone architecture")
    parser.add_argument('--feat-dim', default=768, type=int, help='feature dimension')
    parser.add_argument('--img-h', default=256, type=int)
    parser.add_argument('--img-w', default=128, type=int)
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument("--save-path", default="save/", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 70])
    parser.add_argument('--sigma', default=0.8, type=float)
    parser.add_argument('-T', '--temperature', default=3, type=float)
    
    parser.add_argument('--batch-pidnum', default=8, type=int)
    parser.add_argument('--pid-numsample', default=4, type=int)
    parser.add_argument('--test-batch', default=128, type=int)
    parser.add_argument('--stage1-epoch', default=20, type=int)
    parser.add_argument('--stage2-epoch', default=120, type=int)
    
    parser.add_argument('--relabel', default=1, type=int)
    parser.add_argument('--weak-weight', default=0.25, type=float)
    parser.add_argument('--tri-weight', default=0.25, type=float)
    parser.add_argument('--debug', default='wsl', type=str)
    
    parser.add_argument('--trial', default=1, type=int)
    parser.add_argument('--search-mode', default='all', type=str)
    parser.add_argument('--gall-mode', default='single', type=str)
    parser.add_argument('--test-mode', default='t2v', type=str)
    
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--model-path', default='default', type=str)
    
    args = parser.parse_args()
    args.save_path = './saved_' + args.dataset + '_' + args.arch + '/' + args.save_path
    
    if args.dataset =='sysu':
        args.num_classes = 395
    elif args.dataset =='regdb':
        args.num_classes = 206
        args.save_path += f'_{args.trial}'
    elif args.dataset == 'llcm':
        args.num_classes = 713
        
    set_seed(args.seed)
    setproctitle.setproctitle(args.save_path)
    main(args)