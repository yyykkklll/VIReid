import os
import argparse
import setproctitle
import torch
import warnings
import time
import copy 

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
        # Phase 1: 
        # 修改: 引入频域增强后，数据效率提升，适当缩短 Warmup
        # ======================================================
        if enable_phase1:
            logger('Time: {} | [Phase 1] Mamba + FreqAug Warmup'.format(time_now()))
            for current_epoch in range(0, args.stage1_epoch):
                model.scheduler_phase1.step(current_epoch)
                
                _, result = train(args, model, dataset, current_epoch, sgm, logger, enable_phase1=True)
                
                cmc, mAP, mINP = test(args, model, dataset, current_epoch) 
                best_rank1 = max(cmc[0], best_rank1)
                best_mAP = max(mAP, best_mAP)
                
                logger('Time: {} | P1 Epoch {}; {}'.format(time_now(), current_epoch+1, result))
                logger('R1:{:.2f} | mAP:{:.2f}'.format(cmc[0]*100, mAP*100))
                
                if current_epoch == args.stage1_epoch - 1:
                    save_checkpoint(args, model, current_epoch + 1)

        # ======================================================
        # Phase 2: 
        # ======================================================
        enable_phase1 = False
        start_epoch = model.resume_epoch
        if start_epoch < args.stage1_epoch and 'wsl' in args.debug and not args.resume:
             start_epoch = args.stage1_epoch

        logger('Time: {} | [Phase 2] Structure-Aware Alignment'.format(time_now()))
        
        # [关键修复] 移除错误的权重覆盖逻辑
        # 原代码: model.classifier3.load_state_dict(model.classifier1.state_dict())
        # 新逻辑: 保持独立初始化，或仅在第一次进入 P2 时做温和的初始化
        # 由于我们现在有了 FreqAug，classifier3 从头学也能很快收敛，因此直接移除该行。
        
        for current_epoch in range(start_epoch, args.stage2_epoch):
            model.scheduler_phase2.step(current_epoch)
            
            _, result = train(args, model, dataset, current_epoch, sgm, logger, enable_phase1=False)

            cmc, mAP, mINP = test(args, model, dataset, current_epoch) 
            
            is_best_rank = (cmc[0] >= best_rank1)
            best_rank1 = max(cmc[0], best_rank1)
            best_mAP = max(mAP, best_mAP)
            
            model.save_model(current_epoch + 1, is_best_rank)
            
            logger('Epoch: {} | LR: {:.6f}'.format(current_epoch + 1, model.scheduler_phase2.get_lr()[0]))
            logger(result)
            logger('R1:{:.2f} | mAP:{:.2f} | Best R1:{:.2f}'.format(cmc[0]*100, mAP*100, best_rank1*100))
        
    if args.mode == 'test':
        model.resume_model(args.model_path)
        cmc, mAP, mINP = test(args, model, dataset)
        logger('R1:{:.2f} | mAP:{:.2f}'.format(cmc[0]*100, mAP*100))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("FD-Mamba: Frequency-Disentangled Mamba for VI-ReID")
    parser.add_argument("--dataset", default="regdb", type=str)
    parser.add_argument("--data-path", default="/root/vireid/datasets", type=str)
    
    # [修改] 默认架构改为 vmamba
    parser.add_argument("--arch", default="vmamba", type=str, help="backbone architecture: vit or vmamba")
    
    parser.add_argument('--feat-dim', default=384, type=int, help='feature dimension (384 for mamba-small)')
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