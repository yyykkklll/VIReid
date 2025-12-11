"""
Cross-Modal Person Re-Identification Training Module
====================================================
Features:
- Phase 1: Intra-modal learning (RGB and IR trained independently)
- Phase 2: Cross-modal matching learning (based on pseudo labels)
- Multiple loss functions: ID Loss, Triplet Loss, CMO Loss, Weak Loss
- CLIP semantic enhancement and Sinkhorn global matching
Author: Fixed Version
Date: 2025-12-11
"""

import torch
import torch.nn.functional as F
from models import Model
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy
from tqdm import tqdm


def train(args, model: Model, dataset, epoch, cma: CMA, logger, enable_phase1):
    """
    Main training function
    
    Args:
        args: Training configuration parameters
        model: Training model
        dataset: Dataset object
        epoch: Current epoch
        cma: Cross-modal matching aggregator
        logger: Logger for recording
        enable_phase1: Whether in Phase 1 stage
        
    Returns:
        loss_dict: Loss dictionary
        loss_str: Loss string
    """
    
    if not enable_phase1:
        logger(f"Epoch {epoch}: Starting cross-modal feature extraction and matching...")
        
        r2i_pair_dict, i2r_pair_dict = cma.extract_and_match(
            model=model,
            dataset=dataset,
            clip_model=model.clip_model if hasattr(model, 'clip_model') else None
        )
        
        match_info = _build_matching_matrices(
            args, r2i_pair_dict, i2r_pair_dict, model.device
        )
        
        logger(f"Matching completed: Common={len(match_info['common_dict'])}, "
               f"Specific={len(match_info['specific_dict'])}, "
               f"Remain={len(match_info['remain_dict'])}")
        
        if not model.enable_cls3:
            model.enable_cls3 = True
    else:
        match_info = None
    
    model.set_train()
    meter = MultiItemAverageMeter()
    
    rgb_loader, ir_loader = dataset.get_train_loader()
    batch_size = args.batch_pidnum * args.pid_numsample
    
    iters = min(len(rgb_loader), len(ir_loader))
    pbar = tqdm(
        zip(rgb_loader, ir_loader), 
        total=iters, 
        desc=f"Epoch {epoch} ({'Phase1' if enable_phase1 else 'Phase2'})",
        leave=False
    )
    
    for (rgb_imgs, ca_imgs, rgb_info), (ir_imgs, aug_imgs, ir_info) in pbar:
        
        rgb_imgs = rgb_imgs.to(model.device)
        ca_imgs = ca_imgs.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        
        if args.dataset == 'regdb':
            aug_imgs = aug_imgs.to(model.device)
            ir_imgs_full = torch.cat([ir_imgs, aug_imgs], dim=0)
        else:
            ir_imgs_full = ir_imgs
        
        color_imgs = torch.cat([rgb_imgs, ca_imgs], dim=0)
        
        rgb_ids = torch.cat([rgb_info[:, 1], rgb_info[:, 1]]).to(model.device)
        ir_ids = ir_info[:, 1].to(model.device)
        if args.dataset == 'regdb':
            ir_ids = torch.cat([ir_ids, ir_ids]).to(model.device)
        
        gap_features, bn_features = model.model(color_imgs, ir_imgs_full)
        
        rgbcls_out, _ = model.classifier1(bn_features)
        ircls_out, _ = model.classifier2(bn_features)
        
        rgb_features = gap_features[:2 * batch_size]
        ir_features = gap_features[2 * batch_size:]
        
        r2r_cls = rgbcls_out[:2 * batch_size]
        i2i_cls = ircls_out[2 * batch_size:]
        r2i_cls = ircls_out[:2 * batch_size]
        i2r_cls = rgbcls_out[2 * batch_size:]
        
        if enable_phase1:
            total_loss, losses = _compute_phase1_loss(
                model, args, 
                r2r_cls, i2i_cls, 
                rgb_features, ir_features,
                rgb_ids, ir_ids
            )
        else:
            total_loss, losses = _compute_phase2_loss(
                model, args, epoch,
                bn_features, gap_features,
                r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                rgb_features, ir_features,
                rgb_ids, ir_ids,
                match_info, cma, logger
            )
        
        meter.update(losses)
        
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=5.0)
        
        if enable_phase1:
            model.optimizer_phase1.step()
        else:
            model.optimizer_phase2.step()
        
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
    
    return meter.get_val(), meter.get_str()


def _build_matching_matrices(args, r2i_dict, i2r_dict, device):
    """
    Build cross-modal matching relationship matrices
    
    Matching types:
        - Common: Bidirectional consistent matching (most reliable)
        - Specific: Unidirectional matching (medium reliability)
        - Remain: Conflicting matching (low reliability)
    
    Args:
        args: Configuration arguments
        r2i_dict: RGB to IR matching dictionary
        i2r_dict: IR to RGB matching dictionary
        device: Device to store tensors
        
    Returns:
        match_info: Dictionary containing all matching information
    """
    
    common_dict = {}
    specific_dict = {}
    remain_dict = {}
    
    for rgb_id, ir_id in r2i_dict.items():
        if ir_id in i2r_dict and i2r_dict[ir_id] == rgb_id:
            common_dict[rgb_id] = ir_id
        elif rgb_id not in i2r_dict.values() and ir_id not in i2r_dict.keys():
            specific_dict[rgb_id] = ir_id
        else:
            remain_dict[rgb_id] = ir_id
    
    for ir_id, rgb_id in i2r_dict.items():
        if (rgb_id, ir_id) not in common_dict.items():
            if rgb_id not in r2i_dict.keys() and ir_id not in r2i_dict.values():
                specific_dict[rgb_id] = ir_id
            else:
                remain_dict[rgb_id] = ir_id
    
    num_classes = args.num_classes
    common_rm = torch.zeros(num_classes, num_classes).to(device)
    specific_rm = torch.zeros(num_classes, num_classes).to(device)
    remain_rm = torch.zeros(num_classes, num_classes).to(device)
    
    for rgb_id, ir_id in common_dict.items():
        common_rm[rgb_id, ir_id] = 1.0
    
    for rgb_id, ir_id in specific_dict.items():
        specific_rm[rgb_id, ir_id] = 1.0
    
    for rgb_id, ir_id in remain_dict.items():
        remain_rm[rgb_id, ir_id] = 1.0
    
    specific_rm = specific_rm + common_rm
    
    common_matched_rgb = torch.tensor(list(common_dict.keys())).to(device) if common_dict else torch.tensor([]).to(device)
    common_matched_ir = torch.tensor(list(common_dict.values())).to(device) if common_dict else torch.tensor([]).to(device)
    specific_matched_rgb = torch.tensor(list(specific_dict.keys())).to(device) if specific_dict else torch.tensor([]).to(device)
    remain_matched_rgb = torch.tensor(list(remain_dict.keys())).to(device) if remain_dict else torch.tensor([]).to(device)
    remain_matched_ir = torch.tensor(list(remain_dict.values())).to(device) if remain_dict else torch.tensor([]).to(device)
    
    return {
        'common_dict': common_dict,
        'specific_dict': specific_dict,
        'remain_dict': remain_dict,
        'common_rm': common_rm,
        'specific_rm': specific_rm,
        'remain_rm': remain_rm,
        'common_matched_rgb': common_matched_rgb,
        'common_matched_ir': common_matched_ir,
        'specific_matched_rgb': specific_matched_rgb,
        'remain_matched_rgb': remain_matched_rgb,
        'remain_matched_ir': remain_matched_ir
    }


def _compute_phase1_loss(model, args, r2r_cls, i2i_cls, 
                         rgb_features, ir_features, rgb_ids, ir_ids):
    """
    Phase 1 Loss: Intra-modal learning
    
    Loss components:
        - ID Loss: Classification loss
        - Triplet Loss: Metric learning loss
    
    Args:
        model: Training model
        args: Configuration arguments
        r2r_cls: RGB to RGB classification scores
        i2i_cls: IR to IR classification scores
        rgb_features: RGB features
        ir_features: IR features
        rgb_ids: RGB identity labels
        ir_ids: IR identity labels
        
    Returns:
        total_loss: Total loss
        losses: Loss dictionary
    """
    
    r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
    i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
    
    r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
    i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
    
    total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
    
    losses = {
        'r2r_id_loss': r2r_id_loss.item(),
        'i2i_id_loss': i2i_id_loss.item(),
        'r2r_tri_loss': r2r_tri_loss.item(),
        'i2i_tri_loss': i2i_tri_loss.item()
    }
    
    return total_loss, losses


def _compute_phase2_loss(model, args, epoch,
                         bn_features, gap_features,
                         r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                         rgb_features, ir_features,
                         rgb_ids, ir_ids,
                         match_info, cma, logger):
    """
    Phase 2 Loss: Cross-modal learning
    
    Loss components:
        - Basic ID Loss (with detached backbone)
        - Cross-modal Triplet Loss
        - CMO Loss (Cross-Modal Consistency)
        - Cross-Modal Classification Loss
        - Weak Supervision Loss (for Remain branch)
    
    Args:
        model: Training model
        args: Configuration arguments
        epoch: Current epoch
        bn_features: Batch normalized features
        gap_features: Global average pooling features
        r2r_cls/i2i_cls/r2i_cls/i2r_cls: Classification scores
        rgb_features/ir_features: Modal features
        rgb_ids/ir_ids: Identity labels
        match_info: Matching information dictionary
        cma: Cross-modal matching aggregator
        logger: Logger
        
    Returns:
        total_loss: Total loss
        losses: Loss dictionary
    """
    
    batch_size = rgb_ids.size(0)
    
    dtd_features = bn_features.detach()
    dtd_rgbcls_out, _ = model.classifier1(dtd_features)
    dtd_ircls_out, _ = model.classifier2(dtd_features)
    
    dtd_r2r_cls = dtd_rgbcls_out[:batch_size]
    dtd_i2i_cls = dtd_ircls_out[batch_size:]
    
    r2r_id_loss = model.pid_criterion(dtd_r2r_cls, rgb_ids)
    i2i_id_loss = model.pid_criterion(dtd_i2i_cls, ir_ids)
    
    total_loss = r2r_id_loss + i2i_id_loss
    losses = {
        'r2r_id_loss': r2r_id_loss.item(),
        'i2i_id_loss': i2i_id_loss.item()
    }
    
    common_rgb_indices = torch.isin(rgb_ids, match_info['common_matched_rgb'])
    common_ir_indices = torch.isin(ir_ids, match_info['common_matched_ir'])
    
    if common_rgb_indices.any() and common_ir_indices.any():
        
        selected_rgb_ids = rgb_ids[common_rgb_indices]
        selected_ir_ids = ir_ids[common_ir_indices]
        
        translated_rgb_label = torch.nonzero(
            match_info['common_rm'][selected_rgb_ids], as_tuple=False
        )[:, -1]
        translated_ir_label = torch.nonzero(
            match_info['common_rm'].T[selected_ir_ids], as_tuple=False
        )[:, -1]
        
        selected_rgb_features = rgb_features[common_rgb_indices]
        selected_ir_features = ir_features[common_ir_indices]
        
        matched_rgb_features = torch.cat([selected_rgb_features, ir_features], dim=0)
        matched_rgb_labels = torch.cat([translated_rgb_label, ir_ids], dim=0)
        
        matched_ir_features = torch.cat([rgb_features, selected_ir_features], dim=0)
        matched_ir_labels = torch.cat([rgb_ids, translated_ir_label], dim=0)
        
        tri_loss_rgb = args.tri_weight * model.tri_criterion(
            matched_rgb_features, matched_rgb_labels
        )
        tri_loss_ir = args.tri_weight * model.tri_criterion(
            matched_ir_features, matched_ir_labels
        )
        
        total_loss += tri_loss_rgb + tri_loss_ir
        losses.update({
            'tri_loss_rgb': tri_loss_rgb.item(),
            'tri_loss_ir': tri_loss_ir.item()
        })
        
        cma._update_memory(
            bn_features[:batch_size], 
            bn_features[batch_size:],
            rgb_ids, 
            ir_ids
        )
        
        r2i_entropy = infoEntropy(r2i_cls).item()
        i2r_entropy = infoEntropy(i2r_cls).item()
        
        total_entropy = max(2.0 - r2i_entropy - i2r_entropy, 0.1)
        w_r2i = max((1.0 - r2i_entropy) / total_entropy, 0.01)
        w_i2r = max((1.0 - i2r_entropy) / total_entropy, 0.01)
        
        selected_rgb_memory = cma.vis_memory[translated_ir_label].detach()
        selected_ir_memory = cma.ir_memory[translated_rgb_label].detach()
        
        if torch.norm(selected_rgb_memory) < 1e-6 or torch.norm(selected_ir_memory) < 1e-6:
            logger(f"Epoch {epoch}: Memory bank not initialized, skipping CMO loss")
        else:
            mem_r2i_cls, _ = model.classifier2(selected_rgb_memory)
            mem_i2r_cls, _ = model.classifier1(selected_ir_memory)
            
            cmo_criterion = torch.nn.MSELoss()
            
            if selected_ir_ids.shape[0] > 0:
                r2i_cmo_loss = w_r2i * cmo_criterion(
                    dtd_i2i_cls[common_ir_indices], 
                    mem_r2i_cls
                )
                if not torch.isnan(r2i_cmo_loss) and not torch.isinf(r2i_cmo_loss):
                    total_loss += r2i_cmo_loss
                    losses['r2i_cmo_loss'] = r2i_cmo_loss.item()
                else:
                    logger(f"Epoch {epoch}: r2i_cmo_loss is NaN/Inf, skipped")
            
            if selected_rgb_ids.shape[0] > 0:
                i2r_cmo_loss = w_i2r * cmo_criterion(
                    dtd_r2r_cls[common_rgb_indices], 
                    mem_i2r_cls
                )
                if not torch.isnan(i2r_cmo_loss) and not torch.isinf(i2r_cmo_loss):
                    total_loss += i2r_cmo_loss
                    losses['i2r_cmo_loss'] = i2r_cmo_loss.item()
                else:
                    logger(f"Epoch {epoch}: i2r_cmo_loss is NaN/Inf, skipped")
    
    if epoch >= 30 and match_info['remain_dict']:
        
        r2c_cls, _ = model.classifier3(bn_features[:batch_size])
        i2c_cls, _ = model.classifier3(bn_features[batch_size:])
        
        remain_rgb_indices = torch.isin(rgb_ids, match_info['remain_matched_rgb'])
        
        if remain_rgb_indices.any():
            remain_rgb_ids = rgb_ids[remain_rgb_indices]
            remain_r2c_cls = r2c_cls[remain_rgb_indices]
            
            weak_r2c_loss = args.weak_weight * model.weak_criterion(
                remain_r2c_cls, 
                match_info['remain_rm'][remain_rgb_ids]
            )
            
            if not torch.isnan(weak_r2c_loss) and not torch.isinf(weak_r2c_loss):
                total_loss += weak_r2c_loss
                losses['weak_r2c_loss'] = weak_r2c_loss.item()
            else:
                logger(f"Epoch {epoch}: weak_r2c_loss is NaN/Inf, skipped")
    
    r2c_cls, _ = model.classifier3(bn_features[:batch_size])
    i2c_cls, _ = model.classifier3(bn_features[batch_size:])
    
    specific_rgb_indices = torch.isin(rgb_ids, match_info['specific_matched_rgb'])
    common_rgb_indices = torch.isin(rgb_ids, match_info['common_matched_rgb'])
    rgb_indices = specific_rgb_indices ^ common_rgb_indices
    
    if rgb_indices.any():
        selected_rgb_ids = rgb_ids[rgb_indices]
        selected_r2c_cls = r2c_cls[rgb_indices]
        
        pseudo_labels = torch.argmax(match_info['specific_rm'][selected_rgb_ids], dim=1)
        rgb_cross_loss = model.pid_criterion(selected_r2c_cls, pseudo_labels)
        
        if not torch.isnan(rgb_cross_loss) and not torch.isinf(rgb_cross_loss):
            total_loss += rgb_cross_loss
            losses['rgb_cross_loss'] = rgb_cross_loss.item()
    
    ir_cross_loss = model.pid_criterion(i2c_cls, ir_ids)
    total_loss += ir_cross_loss
    losses['ir_cross_loss'] = ir_cross_loss.item()
    
    return total_loss, losses
