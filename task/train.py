import torch
from utils import MultiItemAverageMeter
from models import Model
import numpy as np
from tqdm import tqdm  # [新增] 引入进度条库

def train(args, model: Model, dataset, current_epoch, sgm, logger, enable_phase1=False):
    """
    Two-Stage Training for SG-WSL
    Phase 1: 单模态预热 (Intra-modal Warmup)
    Phase 2: 结构感知对齐 (Structure-Aware Alignment)
    """
    meter = MultiItemAverageMeter()
    model.set_train()
    
    # ======================================================
    # Phase 2 准备工作: 挖掘跨模态关系
    # ======================================================
    common_rgb_tensor = None
    common_ir_tensor = None
    
    if not enable_phase1:
        # 1. 提取全量特征
        sgm.extract(args, model, dataset) 
        # 提取完后记得切回训练模式
        model.set_train() 
        
        # 2. 挖掘结构化关系 (Mining Structure Relations)
        r2i_pair_dict, i2r_pair_dict = sgm.mining_structure_relations(current_epoch)
        
        # 3. 交叉验证一致性 (Cross-Check Consistency)
        common_dict = {}
        remain_dict = {}
        
        for r, i in r2i_pair_dict.items():
            if i in i2r_pair_dict and i2r_pair_dict[i] == r:
                common_dict[r] = i
            else:
                remain_dict[r] = i 
        
        common_rgb_pids = list(common_dict.keys())
        common_ir_pids = list(common_dict.values())
        
        # 转为 Tensor 方便后续在 Batch 中快速索引
        if len(common_rgb_pids) > 0:
            common_rgb_tensor = torch.tensor(common_rgb_pids).to(model.device)
            common_ir_tensor = torch.tensor(common_ir_pids).to(model.device)
        else:
            common_rgb_tensor = torch.tensor([]).to(model.device)
            common_ir_tensor = torch.tensor([]).to(model.device)
            
        # 激活共享分类器 (Phase 2 开始训练 Classifier3)
        if not model.enable_cls3:
            model.enable_cls3 = True

    # ======================================================
    # 数据加载与 Batch 训练
    # ======================================================
    rgb_loader, ir_loader = dataset.get_train_loader()
    
    # [新增] 使用 tqdm 包装 loader
    # ncols=120 控制进度条宽度，避免太长换行
    pbar = tqdm(zip(rgb_loader, ir_loader), total=len(rgb_loader), 
                desc=f"Epoch {current_epoch+1}/{args.stage2_epoch}", ncols=120)
    
    for batch_idx, ((rgb_imgs, ca_imgs, rgb_info), (ir_imgs, aug_imgs, ir_info)) in enumerate(pbar):
        
        # 优化器梯度清零
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()

        # 数据移至 GPU
        rgb_imgs = rgb_imgs.to(model.device)
        ca_imgs = ca_imgs.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        aug_imgs = aug_imgs.to(model.device)
        
        all_rgb = torch.cat((rgb_imgs, ca_imgs), dim=0)
        all_ir = torch.cat((ir_imgs, aug_imgs), dim=0)
        
        # 标签处理
        rgb_pids = rgb_info[:, 1].to(model.device)
        ir_pids = ir_info[:, 1].to(model.device)
        
        rgb_pids = torch.cat((rgb_pids, rgb_pids), dim=0)
        ir_pids = torch.cat((ir_pids, ir_pids), dim=0)

        # ======================================================
        # Forward Pass
        # ======================================================
        # ViT 同时处理 RGB 和 IR 数据
        gap_features, bn_features = model.model(all_rgb, all_ir)
        
        # 拆分特征
        total_len = gap_features.size(0)
        half_len = total_len // 2
        rgb_gap = gap_features[:half_len]
        ir_gap = gap_features[half_len:]
        rgb_bn = bn_features[:half_len]
        ir_bn = bn_features[half_len:]

        # 专家分类器预测 (Heterogeneous Experts)
        rgb_score, _ = model.classifier1(rgb_bn) 
        ir_score, _ = model.classifier2(ir_bn)   

        total_loss = 0.0

        # ======================================================
        # Phase 1: 单模态预热 (Intra-modal Warmup)
        # ======================================================
        # 仅使用单模态内的 ID Loss 和 Triplet Loss
        loss_id_rgb = model.pid_criterion(rgb_score, rgb_pids)
        loss_tri_rgb = model.tri_criterion(rgb_gap, rgb_pids)
        
        loss_id_ir = model.pid_criterion(ir_score, ir_pids)
        loss_tri_ir = model.tri_criterion(ir_gap, ir_pids)
        
        total_loss = loss_id_rgb + loss_id_ir + \
                     args.tri_weight * (loss_tri_rgb + loss_tri_ir)
        
        meter.update({
            'p1_id_rgb': loss_id_rgb.item(), 
            'p1_id_ir': loss_id_ir.item(),
            'p1_tri': (loss_tri_rgb + loss_tri_ir).item()
        })

        # ======================================================
        # Phase 2: 结构感知协同学习 (Structure-Aware Consistency)
        # ======================================================
        if not enable_phase1:
            # 1. 共享分类器预测 (Shared Knowledge)
            rgb_shared_score, _ = model.classifier3(rgb_bn)
            ir_shared_score, _ = model.classifier3(ir_bn)
            
            # 2. 一致性损失 (Consistency Loss)
            if common_rgb_tensor is not None and len(common_rgb_tensor) > 0:
                rgb_is_common = torch.isin(rgb_pids, common_rgb_tensor)
                ir_is_common = torch.isin(ir_pids, common_ir_tensor)
                
                if rgb_is_common.any() and ir_is_common.any():
                    loss_id_shared = model.pid_criterion(rgb_shared_score[rgb_is_common], rgb_pids[rgb_is_common]) + \
                                     model.pid_criterion(ir_shared_score[ir_is_common], ir_pids[ir_is_common])
                    
                    total_loss += 0.5 * loss_id_shared
                    meter.update({'p2_id_shared': loss_id_shared.item()})

            # 3. 弱监督损失 (Weak Loss)
            if current_epoch >= args.stage1_epoch:
                 pseudo_label_rgb = rgb_score.detach().argmax(dim=1)
                 pseudo_label_ir = ir_score.detach().argmax(dim=1)
                 
                 loss_weak_rgb = args.weak_weight * model.weak_criterion(rgb_shared_score, pseudo_label_rgb)
                 loss_weak_ir = args.weak_weight * model.weak_criterion(ir_shared_score, pseudo_label_ir)
                 
                 total_loss += loss_weak_rgb + loss_weak_ir
                 meter.update({'p2_weak': (loss_weak_rgb + loss_weak_ir).item()})

        # ======================================================
        # 反向传播
        # ======================================================
        total_loss.backward()
        
        if enable_phase1:
            model.optimizer_phase1.step()
        else:
            model.optimizer_phase2.step()
            
        # [新增] 实时更新进度条信息
        pbar.set_postfix({
            'Loss': '{:.4f}'.format(total_loss.item()),
            'P1_ID': '{:.2f}'.format(loss_id_rgb.item() + loss_id_ir.item())
        })

    return meter.get_val(), meter.get_str()