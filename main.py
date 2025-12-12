import os
import argparse
import setproctitle
import torch
import warnings

import time
import datasets
import models
from task import train, test
from wsl import CMA
from utils import time_now, makedir, Logger, MultiItemAverageMeter, set_seed, save_checkpoint

warnings.filterwarnings("ignore")

def main(args):
    best_rank1 = 0
    best_mAP = 0
    best_epoch = 0
    patience_counter = 0
    max_patience = 15  # 早停耐心值
    
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
        
        # 加载预训练模型
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

        # ==================== Phase 1 Training ====================
        if enable_phase1:
            logger('=' * 60)
            logger('Time: {} | Starting Phase 1 from epoch 0'.format(time_now()))
            logger('=' * 60)
            
            for current_epoch in range(0, args.stage1_epoch):
                model.scheduler_phase1.step()
                
                # 训练
                _, result = train(args, model, dataset, current_epoch, cma, logger, enable_phase1)
                
                # 测试
                cmc, mAP, mINP = test(args, model, dataset, current_epoch)
                
                # ========== 异常检测 ==========
                if current_epoch > 5:  # 前5个epoch允许不稳定
                    # 检测性能崩溃（R1下降超过50%或低于10%）
                    if cmc[0] < best_rank1 * 0.5 or cmc[0] < 0.10:
                        logger('=' * 60)
                        logger('⚠️ WARNING: Performance collapse detected!')
                        logger(f'   Current R1: {cmc[0]:.4f} | Best R1: {best_rank1:.4f}')
                        logger(f'   Drop: {(best_rank1 - cmc[0]) / best_rank1 * 100:.1f}%')
                        logger('   Action: Skipping this epoch, model not saved')
                        logger('=' * 60)
                        patience_counter += 1
                        
                        # 如果连续3次崩溃，降低学习率
                        if patience_counter >= 3:
                            logger('⚠️ Multiple collapses detected, reducing learning rate by 50%')
                            for param_group in model.optimizer_phase1.param_groups:
                                param_group['lr'] *= 0.5
                            patience_counter = 0
                        continue
                # ==============================
                
                # 更新最佳指标
                is_best_rank = (cmc[0] >= best_rank1)
                is_best_mAP = (mAP >= best_mAP)
                
                if is_best_rank:
                    best_rank1 = cmc[0]
                    best_epoch = current_epoch + 1
                    patience_counter = 0
                    logger(f'✓ New best R1: {best_rank1:.4f} at epoch {best_epoch}')
                else:
                    patience_counter += 1
                
                if is_best_mAP:
                    best_mAP = mAP
                
                # ========== 早停检测 ==========
                if patience_counter >= max_patience:
                    logger('=' * 60)
                    logger(f'Early stopping triggered at epoch {current_epoch + 1}')
                    logger(f'Best R1: {best_rank1:.4f} at epoch {best_epoch}')
                    logger(f'Best mAP: {best_mAP:.4f}')
                    logger('=' * 60)
                    break
                # ==============================
                
                # 日志输出
                logger('=' * 60)
                logger('Epoch: {} | Time: {} | Setting: {}'.format(
                    current_epoch + 1, time_now(), args.save_path))
                logger(f'Learning Rate: {model.scheduler_phase1.get_last_lr()[0]:.2e}')
                logger(result)
                logger('R1: {:.4f} | R10: {:.4f} | R20: {:.4f} | mAP: {:.4f} | mINP: {:.4f}'.format(
                    cmc[0], cmc[9], cmc[19], mAP, mINP))
                logger('Best R1: {:.4f} (Epoch {}) | Best mAP: {:.4f}'.format(
                    best_rank1, best_epoch, best_mAP))
                logger('Patience: {}/{}'.format(patience_counter, max_patience))
                logger('=' * 60)
                
                # 定期保存检查点
                if (current_epoch + 1) % 10 == 0 or current_epoch == args.stage1_epoch - 1:
                    save_checkpoint(args, model, current_epoch + 1)
                    logger(f'✓ Checkpoint saved at epoch {current_epoch + 1}')
        
        # ==================== Phase 2 Training ====================
        enable_phase1 = False
        start_epoch = model.resume_epoch
        patience_counter = 0  # 重置耐心计数器
        
        logger('=' * 60)
        logger('Time: {} | Starting Phase 2 from epoch {}'.format(time_now(), start_epoch))
        logger('=' * 60)
        
        for current_epoch in range(start_epoch, args.stage2_epoch):
            model.scheduler_phase2.step()
            
            # 训练
            _, result = train(args, model, dataset, current_epoch, cma, logger, enable_phase1)
            
            # 测试
            cmc, mAP, mINP = test(args, model, dataset, current_epoch)
            
            # ========== 异常检测 ==========
            if current_epoch > start_epoch + 5:
                if cmc[0] < best_rank1 * 0.5 or cmc[0] < 0.15:
                    logger('=' * 60)
                    logger('⚠️ WARNING: Performance collapse detected!')
                    logger(f'   Current R1: {cmc[0]:.4f} | Best R1: {best_rank1:.4f}')
                    logger(f'   Drop: {(best_rank1 - cmc[0]) / best_rank1 * 100:.1f}%')
                    logger('   Action: Skipping this epoch, model not saved')
                    logger('=' * 60)
                    patience_counter += 1
                    
                    if patience_counter >= 3:
                        logger('⚠️ Multiple collapses detected, reducing learning rate by 50%')
                        for param_group in model.optimizer_phase2.param_groups:
                            param_group['lr'] *= 0.5
                        patience_counter = 0
                    continue
            # ==============================
            
            # 更新最佳指标
            is_best_rank = (cmc[0] >= best_rank1)
            is_best_mAP = (mAP >= best_mAP)
            
            if is_best_rank:
                best_rank1 = cmc[0]
                best_epoch = current_epoch + 1
                patience_counter = 0
                logger(f'✓ New best R1: {best_rank1:.4f} at epoch {best_epoch}')
            else:
                patience_counter += 1
            
            if is_best_mAP:
                best_mAP = mAP
            
            # 保存最佳模型
            if is_best_rank:
                model.save_model(current_epoch + 1, is_best_rank)
            
            # ========== 早停检测 ==========
            if patience_counter >= max_patience:
                logger('=' * 60)
                logger(f'Early stopping triggered at epoch {current_epoch + 1}')
                logger(f'Best R1: {best_rank1:.4f} at epoch {best_epoch}')
                logger(f'Best mAP: {best_mAP:.4f}')
                logger('=' * 60)
                break
            # ==============================
            
            # 日志输出
            logger('=' * 60)
            logger('Epoch: {} | Time: {} | Setting: {}'.format(
                current_epoch + 1, time_now(), args.save_path))
            logger(f'Learning Rate: {model.scheduler_phase2.get_last_lr()[0]:.2e}')
            logger(result)
            logger('R1: {:.4f} | R10: {:.4f} | R20: {:.4f} | mAP: {:.4f} | mINP: {:.4f}'.format(
                cmc[0], cmc[9], cmc[19], mAP, mINP))
            logger('Best R1: {:.4f} (Epoch {}) | Best mAP: {:.4f}'.format(
                best_rank1, best_epoch, best_mAP))
            logger('Patience: {}/{}'.format(patience_counter, max_patience))
            logger('=' * 60)
        
        # 训练完成总结
        logger('=' * 60)
        logger('Training Completed!')
        logger(f'Best R1: {best_rank1:.4f} at epoch {best_epoch}')
        logger(f'Best mAP: {best_mAP:.4f}')
        logger('=' * 60)
        
    # ==================== Test Mode ====================
    if args.mode == 'test':
        if args.model_path == 'default':
            model.resume_model()
        else:
            model.resume_model(args.model_path)
        
        cmc, mAP, mINP = test(args, model, dataset)
        
        logger('=' * 60)
        logger('Test Results on Dataset: {}'.format(args.dataset))
        logger('Time: {}'.format(time_now()))
        logger('=' * 60)
        logger('R1: {:.4f} | R10: {:.4f} | R20: {:.4f}'.format(cmc[0], cmc[9], cmc[19]))
        logger('mAP: {:.4f} | mINP: {:.4f}'.format(mAP, mINP))
        logger('=' * 60)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSL-ReID with Feature Diffusion Bridge")
    
    # ==================== Basic Settings ====================
    parser.add_argument("--dataset", default="regdb", type=str, 
                       help="dataset name: sysu, llcm, regdb")
    parser.add_argument("--arch", default="resnet", type=str, 
                       help="architecture: resnet, clip-resnet")
    parser.add_argument('--mode', default='train', type=str,
                       help='train or test')
    parser.add_argument("--data-path", default="./datasets/", type=str, 
                       help="dataset path")
    parser.add_argument("--save-path", default="save/", type=str, 
                       help="log and model save path")
    
    # ==================== Training Settings ====================
    parser.add_argument('--lr', default=0.0003, type=float, 
                       help='learning rate: 0.0003 for sysu/llcm, 0.00045 for regdb')
    parser.add_argument('--weight-decay', default=0.0005, type=float, 
                       help='weight decay')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 70],
                       help='milestones for learning rate decay')
    parser.add_argument('--relabel', default=1, type=int, 
                       help='relabel train dataset')
    parser.add_argument('--weak-weight', default=0.25, type=float, 
                       help='weight of weak loss')
    parser.add_argument('--tri-weight', default=0.25, type=float, 
                       help='weight of triplet loss')
    
    # ==================== Data Settings ====================
    parser.add_argument('--img-h', default=288, type=int,
                       help='height of input image')
    parser.add_argument('--img-w', default=144, type=int,
                       help='width of input image')
    parser.add_argument("--seed", default=1, type=int, 
                       help="random seed")
    parser.add_argument('--num-workers', default=8, type=int, 
                       help='number of workers for dataloader')
    parser.add_argument('--batch-pidnum', default=8, type=int,
                       help='training pid in each batch')
    parser.add_argument('--pid-numsample', default=4, type=int,
                       help='number of samples per pid in a batch')
    parser.add_argument('--test-batch', default=128, type=int, 
                       help='testing batch size')
    
    # ==================== Cross-Modal Settings ====================
    parser.add_argument('--sigma', default=0.8, type=float, 
                       help='momentum update factor for CMA')
    parser.add_argument('-T', '--temperature', default=3, type=float, 
                       help='temperature parameter for softmax')
    
    # ==================== Training Phases ====================
    parser.add_argument("--device", default=0, type=int, 
                       help="gpu device id")
    parser.add_argument('--stage1-epoch', default=20, type=int,
                       help='phase 1 epochs: sysu=20, llcm=80, regdb=50')
    parser.add_argument('--stage2-epoch', default=120, type=int,
                       help='phase 2 total epochs')
    parser.add_argument('--resume', default=0, type=int, 
                       help='resume training from checkpoint')
    parser.add_argument('--debug', default='wsl', type=str,
                       help='training mode: wsl, sl, baseline')
    
    # ==================== Dataset Specific ====================
    parser.add_argument('--trial', default=1, type=int,
                       help='trial id for regdb (1-10)')
    parser.add_argument('--search-mode', default='all', type=str,
                       help='search mode: all or indoor')
    parser.add_argument('--gall-mode', default='single', type=str,
                       help='gallery mode: single or multi')
    parser.add_argument('--test-mode', default='t2v', type=str,
                       help='test mode for regdb/llcm: t2v or v2t')
    parser.add_argument('--model-path', default='default', type=str, 
                       help='path to load pretrained model')
    
    # ==================== Feature Diffusion Bridge ====================
    parser.add_argument('--use-diffusion', default=False, action='store_true',
                       help='enable feature diffusion bridge')
    parser.add_argument('--diffusion-steps', default=10, type=int,
                       help='number of diffusion steps T')
    parser.add_argument('--diffusion-hidden', default=1024, type=int,
                       help='hidden dimension of denoising MLP')
    parser.add_argument('--diffusion-weight', default=0.1, type=float,
                       help='weight of diffusion loss')
    parser.add_argument('--diffusion-lr', default=0.0003, type=float,
                       help='learning rate for diffusion module')
    parser.add_argument('--confidence-weight', default=0.1, type=float,
                       help='weight of confidence guidance loss')
    
    args = parser.parse_args()
    
    # ==================== Post-processing ====================
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
