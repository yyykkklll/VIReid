import torch
from utils import MultiItemAverageMeter
from models import Model
import numpy as np
from tqdm import tqdm
from datasets.data_process import frequency_mix # [新增] 导入频域混合函数

def train(args, model: Model, dataset, current_epoch, sgm, logger, enable_phase1=False):
    """
    Revised Training for Mamba + FreqAug
    """
    meter = MultiItemAverageMeter()
    model.set_train()
    
    # 挖掘跨模态关系 (Phase 2)
    common_rgb_tensor = None
    common_ir_tensor = None
    
    if not enable_phase1:
        # SGM Mining Logic (保持不变)
        sgm.extract(args, model, dataset) 
        model.set_train() 
        r2i_pair_dict, i2r_pair_dict = sgm.mining_structure_relations(current_epoch)
        common_dict = {}
        for r, i in r2i_pair_dict.items():
            if i in i2r_pair_dict and i2r_pair_dict[i] == r:
                common_dict[r] = i
        
        common_rgb_pids = list(common_dict.keys())
        common_ir_pids = list(common_dict.values())
        
        if len(common_rgb_pids) > 0:
            common_rgb_tensor = torch.tensor(common_rgb_pids).to(model.device)
            common_ir_tensor = torch.tensor(common_ir_pids).to(model.device)
        else:
            common_rgb_tensor = torch.tensor([]).to(model.device)
            common_ir_tensor = torch.tensor([]).to(model.device)
            
        if not model.enable_cls3:
            model.enable_cls3 = True

    rgb_loader, ir_loader = dataset.get_train_loader()
    pbar = tqdm(zip(rgb_loader, ir_loader), total=len(rgb_loader), 
                desc=f"Epoch {current_epoch+1}/{args.stage2_epoch}", ncols=120)
    
    for batch_idx, ((rgb_imgs, ca_imgs, rgb_info), (ir_imgs, aug_imgs, ir_info)) in enumerate(pbar):
        
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()

        # 1. 准备基础数据
        rgb_imgs = rgb_imgs.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        # 标签
        rgb_pids = rgb_info[:, 1].to(model.device)
        ir_pids = ir_info[:, 1].to(model.device)

        # 2. [Scheme C] 频域增强 (Frequency Augmentation)
        # 生成 "伪 IR" (RGB content, IR style) 和 "伪 RGB" (IR content, RGB style)
        # 仅在训练阶段的一定概率下或始终开启
        with torch.no_grad():
            mix_rgb, mix_ir = frequency_mix(rgb_imgs, ir_imgs, ratio=0.1)
            mix_rgb = mix_rgb.to(model.device)
            mix_ir = mix_ir.to(model.device)

        # 3. 拼接输入 (Data Efficiency: 一次 Forward 包含 原图 + 增强图)
        # 输入顺序: [RGB, IR, Mix_RGB, Mix_IR]
        # 注意: model.forward 接收 (x1, x2)，这里我们把 RGB 和 Mix_RGB 视作 x1, IR 和 Mix_IR 视作 x2
        
        full_rgb = torch.cat([rgb_imgs, mix_rgb], dim=0) # [2B, 3, H, W]
        full_ir = torch.cat([ir_imgs, mix_ir], dim=0)    # [2B, 3, H, W]
        
        # 标签也需要拼接 (Mix 图的 ID 不变)
        full_rgb_pids = torch.cat([rgb_pids, rgb_pids], dim=0)
        full_ir_pids = torch.cat([ir_pids, ir_pids], dim=0)

        # 4. Forward Pass (Mamba 双流交互)
        gap_features, bn_features = model.model(full_rgb, full_ir)
        
        # 拆分特征
        # 维度: [4B, D] -> RGB(B), MixRGB(B), IR(B), MixIR(B)
        B = rgb_imgs.size(0)
        
        feat_rgb = bn_features[:B]
        feat_mix_rgb = bn_features[B:2*B]
        feat_ir = bn_features[2*B:3*B]
        feat_mix_ir = bn_features[3*B:]
        
        gap_rgb = gap_features[:B]
        gap_ir = gap_features[2*B:3*B]

        # 5. 计算损失
        total_loss = 0.0
        
        # 5.1 专家分类器 Loss (RGB Expert & IR Expert)
        # 只对真实图像计算 ID Loss，保证 Expert 纯净
        rgb_score, _ = model.classifier1(feat_rgb)
        ir_score, _ = model.classifier2(feat_ir)
        
        loss_id_rgb = model.pid_criterion(rgb_score, rgb_pids)
        loss_id_ir = model.pid_criterion(ir_score, ir_pids)
        loss_tri = args.tri_weight * (model.tri_criterion(gap_rgb, rgb_pids) + \
                                      model.tri_criterion(gap_ir, ir_pids))
        
        total_loss += loss_id_rgb + loss_id_ir + loss_tri
        
        meter.update({'id_rgb': loss_id_rgb.item(), 'id_ir': loss_id_ir.item(), 'tri': loss_tri.item()})

        # 5.2 [Scheme C] 一致性损失 (Consistency Loss)
        # 约束: 原图特征 与 风格迁移后特征 距离要近 (结构不变性)
        loss_cons_rgb = model.cons_criterion(feat_rgb, feat_mix_rgb)
        loss_cons_ir = model.cons_criterion(feat_ir, feat_mix_ir)
        
        # 权重设为 0.1~0.5 之间
        loss_cons = 0.2 * (loss_cons_rgb + loss_cons_ir)
        total_loss += loss_cons
        meter.update({'cons': loss_cons.item()})

        # 5.3 Phase 2 Shared Logic
        if not enable_phase1:
            # 共享分类器预测
            rgb_shared_score, _ = model.classifier3(feat_rgb)
            ir_shared_score, _ = model.classifier3(feat_ir)
            
            if common_rgb_tensor is not None and len(common_rgb_tensor) > 0:
                rgb_is_common = torch.isin(rgb_pids, common_rgb_tensor)
                ir_is_common = torch.isin(ir_pids, common_ir_tensor)
                
                if rgb_is_common.any() and ir_is_common.any():
                    loss_id_shared = model.pid_criterion(rgb_shared_score[rgb_is_common], rgb_pids[rgb_is_common]) + \
                                     model.pid_criterion(ir_shared_score[ir_is_common], ir_pids[ir_is_common])
                    total_loss += 0.5 * loss_id_shared
                    meter.update({'p2_shared': loss_id_shared.item()})

        total_loss.backward()
        
        if enable_phase1:
            model.optimizer_phase1.step()
        else:
            model.optimizer_phase2.step()
            
        pbar.set_postfix({
            'Loss': '{:.3f}'.format(total_loss.item()),
            'Cons': '{:.3f}'.format(loss_cons.item())
        })

    return meter.get_val(), meter.get_str()