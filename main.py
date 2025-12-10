import argparse
import os
import torch
import datasets
import models
from task import train, test
from utils import Logger, makedir, set_seed

def main(args):
    log_path = os.path.join(args.save_path, "log/")
    makedir(log_path)
    makedir(os.path.join(args.save_path, "models/"))
    
    logger = Logger(os.path.join(log_path, "log.txt"))
    logger(args)
    
    dataset = datasets.create(args)
    model = models.create(args) # 返回 UnsupervisedModel

    if args.mode == "train":
        best_rank1 = 0
        for epoch in range(args.epochs):
            _, result_str = train(args, model, dataset, epoch, logger)
            logger(f"Epoch {epoch+1}: {result_str}")
            
            if (epoch + 1) % 10 == 0:
                cmc, mAP, mINP = test(args, model, dataset, epoch)
                logger(f"Test Epoch {epoch+1}: R1:{cmc[0]*100:.2f} mAP:{mAP*100:.2f}")
                
                if cmc[0] > best_rank1:
                    best_rank1 = cmc[0]
                    model.save_model(epoch+1, is_best=True)
                model.save_model(epoch+1, is_best=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("USL VI-ReID")
    parser.add_argument("--dataset", default="regdb", type=str)
    parser.add_argument("--data-path", default="./datasets", type=str)
    parser.add_argument("--arch", default="resnet50", type=str)
    parser.add_argument('--img-h', default=256, type=int)
    parser.add_argument('--img-w', default=128, type=int)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument("--save-path", default="save_usl/", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    
    # 训练参数
    parser.add_argument('--lr', default=0.00035, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70])
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch-pidnum', default=8, type=int)
    parser.add_argument('--pid-numsample', default=4, type=int)
    parser.add_argument('--test-batch', default=64, type=int)
    
    # 无监督参数
    parser.add_argument('--lambda-ot', default=0.1, type=float, help="Sinkhorn loss weight")
    parser.add_argument('--lambda-adv', default=0.1, type=float, help="Adversarial loss weight")
    
    # 兼容参数 (dataset需要)
    parser.add_argument('--relabel', default=1, type=int)
    parser.add_argument('--trial', default=1, type=int)
    parser.add_argument('--test-mode', default='t2v', type=str)
    parser.add_argument('--search-mode', default='all', type=str)
    parser.add_argument('--gall-mode', default='single', type=str)
    parser.add_argument('--resume', default=0, type=int)
    
    args = parser.parse_args()
    
    # Num Classes 修正 (用于Dataset初始化，虽然训练不用)
    if args.dataset =='sysu': args.num_classes = 395
    elif args.dataset =='regdb': args.num_classes = 206
    elif args.dataset == 'llcm': args.num_classes = 713
        
    set_seed(args.seed)
    main(args)