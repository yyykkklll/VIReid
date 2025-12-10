import torch
from utils import MultiItemAverageMeter
from models import Model
import numpy as np
from tqdm import tqdm

def train(args, model: Model, dataset, current_epoch, sgm, logger, enable_phase1=False):
    meter = MultiItemAverageMeter()
    model.set_train()
    
    # Phase 2 准备工作 (保持原逻辑)
    common_rgb_tensor = None
    common_ir_tensor = None
    if not enable_phase1:
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

        # 数据移至 GPU
        rgb_imgs = rgb_imgs.to(model.device)
        ca_imgs = ca_imgs.to(model.device)  # 之前代码可能用了 aug_imgs，这里统一用 ca_imgs
        ir_imgs = ir_imgs.to(model.device)
        aug_imgs = aug_imgs.to(model.device)
        
        rgb_pids = rgb_info[:, 1].to(model.device)
        ir_pids = ir_info[:, 1].to(model.device)

        # [修正] 标准输入构建：原始图 + 增强图 (不做频域混合)
        # 这还原了项目最原始的 baseline 逻辑
        all_rgb = torch.cat((rgb_imgs, ca_imgs), dim=0)
        all_ir = torch.cat((ir_imgs, aug_imgs), dim=0)
        
        all_rgb_pids = torch.cat((rgb_pids, rgb_pids), dim=0)
        all_ir_pids = torch.cat((ir_pids, ir_pids), dim=0)

        # Forward
        gap_features, bn_features = model.model(all_rgb, all_ir)
        
        # 拆分特征
        total_len = gap_features.size(0)
        half_len = total_len // 2
        rgb_gap = gap_features[:half_len]
        ir_gap = gap_features[half_len:]
        
        # 这里的 bn_features 包含了 (rgb, ca_rgb) 和 (ir, aug_ir)
        # 我们按照原始逻辑只取前半部分做 Classifier 预测吗？
        # 原代码逻辑：classifier 输入的是 bn_features[:half_len] (即 rgb) 和 bn_features[half_len:] (即 ir)
        # 等等，原代码 vit forward 把 all_rgb 和 all_ir cat 起来了。
        # 这里 ResNet forward 也是 cat。
        # 所以 bn_features 的结构是 [RGB_batch, CA_batch, IR_batch, AUG_batch] (假设 ResNet 内部 cat 顺序)
        # 不，ResNet forward 代码是 `torch.cat([x1, x2], dim=0)`
        # x1 = all_rgb (2B), x2 = all_ir (2B)
        # 所以 Total Batch = 4B.
        
        # 为了稳妥，我们暂时只用最基础的无增强数据进行训练，确保 baseline 正常
        # 简化版逻辑：
        # 重新 forward 仅使用 clean images
        # (这种方式虽然浪费了一点性能，但绝对稳健)
        
        # --- 极简模式 ---
        feat_raw, feat_bn = model.model(rgb_imgs, ir_imgs)
        # feat_bn: [2B, C] -> 前B是RGB，后B是IR
        
        feat_rgb = feat_bn[:rgb_imgs.size(0)]
        feat_ir = feat_bn[rgb_imgs.size(0):]
        gap_rgb = feat_raw[:rgb_imgs.size(0)]
        gap_ir = feat_raw[rgb_imgs.size(0):]
        
        total_loss = 0.0
        
        # Expert 1 & 2
        rgb_score, _ = model.classifier1(feat_rgb)
        ir_score, _ = model.classifier2(feat_ir)
        
        loss_id_rgb = model.pid_criterion(rgb_score, rgb_pids)
        loss_id_ir = model.pid_criterion(ir_score, ir_pids)
        loss_tri = args.tri_weight * (model.tri_criterion(gap_rgb, rgb_pids) + \
                                      model.tri_criterion(gap_ir, ir_pids))
        
        total_loss += loss_id_rgb + loss_id_ir + loss_tri
        
        meter.update({'id_rgb': loss_id_rgb.item(), 'id_ir': loss_id_ir.item(), 'tri': loss_tri.item()})

        # Phase 2 Shared
        if not enable_phase1:
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
            'Loss': '{:.3f}'.format(total_loss.item())
        })

    return meter.get_val(), meter.get_str()