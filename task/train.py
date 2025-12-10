import torch
import numpy as np
from tqdm import tqdm
from utils import MultiItemAverageMeter

def train(args, model, dataset, epoch, logger):
    meter = MultiItemAverageMeter()
    model.set_train()
    
    # 获取 DataLoader
    rgb_loader, ir_loader = dataset.get_train_loader()
    
    pbar = tqdm(zip(rgb_loader, ir_loader), total=len(rgb_loader), 
                desc=f"Epoch {epoch+1}/{args.epochs}", ncols=120)
    
    for batch_idx, ((rgb_imgs, rgb_aug, _), (ir_imgs, ir_aug, _)) in enumerate(pbar):
        rgb_imgs = rgb_imgs.to(model.device)
        rgb_aug = rgb_aug.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        ir_aug = ir_aug.to(model.device)
        
        # -----------------------------------------------------
        # 1. 模态判别器训练 (Discriminator Step)
        # -----------------------------------------------------
        # 提取特征 (不追踪梯度)
        with torch.no_grad():
            # [修正] 去掉 .model，直接调用 backbone 实例
            feat_rgb_det, _ = model.backbone(x1=rgb_imgs)
            feat_ir_det, _ = model.backbone(x2=ir_imgs)
            
        # 判别器预测
        pred_rgb = model.discriminator(feat_rgb_det.detach())
        pred_ir = model.discriminator(feat_ir_det.detach())
        
        # 标签：RGB=1, IR=0
        label_rgb = torch.ones_like(pred_rgb)
        label_ir = torch.zeros_like(pred_ir)
        
        loss_disc = (model.criterion_adv(pred_rgb, label_rgb) + 
                     model.criterion_adv(pred_ir, label_ir)) * 0.5
        
        model.opt_disc.zero_grad()
        loss_disc.backward()
        model.opt_disc.step()
        
        meter.update({'loss_disc': loss_disc.item()})

        # -----------------------------------------------------
        # 2. Backbone & Projector 训练 (Generator Step)
        # -----------------------------------------------------
        # 前向传播 (带梯度)
        # [修正] 去掉 .model，直接调用 backbone 实例
        # Backbone 返回 (gap_feat, bn_feat)，我们通常用 gap_feat 做对比学习
        feat_rgb, _ = model.backbone(x1=rgb_imgs)
        feat_rgb_aug, _ = model.backbone(x1=rgb_aug)
        feat_ir, _ = model.backbone(x2=ir_imgs)
        feat_ir_aug, _ = model.backbone(x2=ir_aug)
        
        # 投影
        proj_rgb = model.projector(feat_rgb)
        proj_rgb_aug = model.projector(feat_rgb_aug)
        proj_ir = model.projector(feat_ir)
        proj_ir_aug = model.projector(feat_ir_aug)
        
        # A. InfoNCE Loss (模态内不变性)
        # RGB vs RGB_Aug
        loss_nce_rgb = model.criterion_nce(torch.cat([proj_rgb, proj_rgb_aug], dim=0))
        # IR vs IR_Aug
        loss_nce_ir = model.criterion_nce(torch.cat([proj_ir, proj_ir_aug], dim=0))
        
        # B. Sinkhorn Loss (跨模态分布对齐)
        # 对齐 RGB 和 IR 的分布
        loss_ot = model.criterion_ot(proj_rgb, proj_ir)
        
        # C. Adversarial Loss (模态混淆)
        # Backbone 希望判别器无法区分 (Label Flip)
        pred_rgb_adv = model.discriminator(feat_rgb)
        pred_ir_adv = model.discriminator(feat_ir)
        
        loss_adv_gen = (model.criterion_adv(pred_rgb_adv, torch.zeros_like(pred_rgb_adv)) + 
                        model.criterion_adv(pred_ir_adv, torch.ones_like(pred_ir_adv))) * 0.5
        
        # 总损失
        total_loss = (loss_nce_rgb + loss_nce_ir) + args.lambda_ot * loss_ot + args.lambda_adv * loss_adv_gen
        
        model.opt_backbone.zero_grad()
        total_loss.backward()
        model.opt_backbone.step()
        
        meter.update({
            'nce_rgb': loss_nce_rgb.item(), 
            'nce_ir': loss_nce_ir.item(),
            'ot': loss_ot.item(),
            'adv_g': loss_adv_gen.item()
        })
        
        pbar.set_postfix({'Loss': '{:.3f}'.format(total_loss.item())})

    return meter.get_val(), meter.get_str()