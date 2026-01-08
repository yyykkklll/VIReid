import torch
from models import Model
from datasets import SYSU
import time
import numpy as np
import random
import copy
from collections import OrderedDict, defaultdict
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy,pha_unwrapping
from models import Model
from models.loss import WeightedContrastiveLoss
from tqdm import tqdm

def train(args, model: Model, dataset, *arg):
    epoch = arg[0]
    cma:CMA = arg[1]
    logger = arg[2]
    enable_phase1 = arg[3]
    debug_logger = arg[4] if len(arg) > 4 else None

    rgb_to_ir_map = defaultdict(list)

    # //get feature for match
    if 'wsl' in args.debug or not enable_phase1:
        cma.extract(args, model, dataset)        
        
        # New Sinkhorn Logic
        match_list = cma.get_label(epoch, debug_logger)
        # match_list: [(rgb_label, ir_label, score), ...]
        
        if match_list:
            for r, i, s in match_list:
                rgb_to_ir_map[r].append((i, s))
                # Since Sinkhorn is symmetric in principle (doubly stochastic), 
                # we can also add ir->rgb map if needed, but we iterate rgb in batch.
        elif not enable_phase1:
             if debug_logger:
                 debug_logger(f"WARNING: No matches found from Sinkhorn for Epoch {epoch} Phase 2!")

        if not model.enable_cls3:
            model.enable_cls3 = True
            
    # Initialize Weighted Loss
    cmcl_criterion = WeightedContrastiveLoss(margin=0.5).to(model.device)

    # ======================================================
    model.set_train()
    meter = MultiItemAverageMeter()
    bt = args.batch_pidnum*args.pid_numsample
    rgb_loader, ir_loader = dataset.get_train_loader()

    nan_batch_counter = 0
    # 创建进度条
    total_batches = min(len(rgb_loader), len(ir_loader))
    phase_name = "Phase1" if enable_phase1 else "Phase2"
    progress_bar = tqdm(zip(rgb_loader, ir_loader), total=total_batches, 
                       desc=f'Epoch {epoch} [{phase_name}]',
                       leave=False)
    
    for batch_idx, ((rgb_imgs, ca_imgs, color_info), (ir_imgs, aug_imgs,ir_info)) in enumerate(progress_bar):
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()
        rgb_imgs, ca_imgs = rgb_imgs.to(model.device), ca_imgs.to(model.device)

        color_imgs = torch.cat((rgb_imgs, ca_imgs), dim = 0)
        rgb_gts, ir_gts = color_info[:,-1], ir_info[:,-1] 
        rgb_ids, ir_ids = color_info[:,1], ir_info[:,1]
        rgb_ids = torch.cat((rgb_ids,rgb_ids)).to(model.device)
        if args.dataset == 'regdb':
            ir_imgs, aug_imgs = ir_imgs.to(model.device), aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim = 0)
            ir_ids = torch.cat((ir_ids,ir_ids)).to(model.device)
        else:
            ir_imgs = ir_imgs.to(model.device)
            ir_ids = ir_ids.to(model.device)
        gap_features, bn_features = model.model(color_imgs, ir_imgs)
        rgbcls_out, _l2_features = model.classifier1(bn_features)
        ircls_out, _l2_features = model.classifier2(bn_features)

        rgb_features, ir_features = gap_features[:2*bt], gap_features[2*bt:]
        r2r_cls, i2i_cls, r2i_cls,i2r_cls \
              = rgbcls_out[:2*bt], ircls_out[2*bt:], ircls_out[:2*bt], rgbcls_out[2*bt:]
        
        total_loss = 0
        
        if 'wsl' in args.debug:
            if enable_phase1:
                r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
                i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
                total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data,
                            'r2r_tri_loss':r2r_tri_loss.data,
                            'i2i_tri_loss':i2i_tri_loss.data})
            else:
                # Phase 2: CMCL + Intra-modal Baseline
                # 1. Intra-modal Loss (Baseline stability)
                r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                meter.update({'r2r_id_loss':r2r_id_loss.data, 'i2i_id_loss':i2i_id_loss.data})
                total_loss += (r2r_id_loss + i2i_id_loss)
                
                # 2. CMCL: Reliability-Aware Loss
                cmcl_loss_val = torch.tensor(0.0, device=model.device)
                valid_pairs = 0
                
                # Iterate over RGB samples in batch
                # rgb_ids has 2*bt elements (original + augmented)
                # rgb_features has 2*bt elements
                
                # We can optimize this loop
                unique_rgb_ids = torch.unique(rgb_ids)
                
                for r_id_tensor in unique_rgb_ids:
                    r_id = r_id_tensor.item()
                    if r_id in rgb_to_ir_map:
                        matches = rgb_to_ir_map[r_id]
                        
                        # Find indices in current RGB batch
                        rgb_indices = (rgb_ids == r_id).nonzero(as_tuple=True)[0]
                        
                        for target_ir_label, score in matches:
                            # Find indices in current IR batch
                            ir_indices = (ir_ids == target_ir_label).nonzero(as_tuple=True)[0]
                            
                            if len(ir_indices) > 0:
                                # Found matches in batch
                                # Form pairs: all rgb instances of r_id vs all ir instances of target_ir_label
                                
                                # rgb_indices: (N_rgb,)
                                # ir_indices: (N_ir,)
                                
                                # Gather features
                                curr_rgb_feats = rgb_features[rgb_indices] # (N_rgb, D)
                                curr_ir_feats = ir_features[ir_indices] # (N_ir, D)
                                
                                # Expand to pairs (N_rgb * N_ir, D)
                                curr_rgb_expanded = curr_rgb_feats.unsqueeze(1).repeat(1, len(ir_indices), 1).view(-1, curr_rgb_feats.size(1))
                                curr_ir_expanded = curr_ir_feats.unsqueeze(0).repeat(len(rgb_indices), 1, 1).view(-1, curr_ir_feats.size(1))
                                
                                scores = torch.tensor([score] * curr_rgb_expanded.size(0), device=model.device)
                                
                                loss = cmcl_criterion(curr_rgb_expanded, curr_ir_expanded, scores)
                                cmcl_loss_val += loss * len(rgb_indices) * len(ir_indices) # Weighted by number of pairs? Or mean? 
                                # WeightedContrastiveLoss returns mean. We accumulate and re-average later.
                                # Actually better to just accumulate the sum of weighted distances
                                
                                valid_pairs += 1

                if valid_pairs > 0:
                     # Since we simply summed means, this might be rough. 
                     # But it captures the gradient.
                    cmcl_loss_val = cmcl_loss_val / valid_pairs
                    meter.update({'cmcl_loss': cmcl_loss_val.data})
                    total_loss += cmcl_loss_val
                
                if debug_logger and (batch_idx % 50 == 0):
                    log_msg = f"Epoch {epoch} Batch {batch_idx}: Valid CMCL Pairs={valid_pairs}"
                    if valid_pairs > 0:
                        log_msg += f", CMCL Loss={cmcl_loss_val.item():.4f}"
                    debug_logger(log_msg)

        elif args.debug == 'baseline':
                r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
                i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
                total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data,
                            'r2r_tri_loss':r2r_tri_loss.data,
                            'i2i_tri_loss':i2i_tri_loss.data})
        
        elif args.debug == 'sl':
            # //supervised learning
            rgb_gts = torch.cat((rgb_gts,rgb_gts)).to(model.device)
            ir_gts = torch.cat((ir_gts,ir_gts)).to(model.device)
            gts = torch.cat((rgb_gts,ir_gts))

            id_loss = model.pid_criterion(rgbcls_out, gts)
            tri_loss = model.tri_criterion(gap_features, gts)
            total_loss = id_loss + args.tri_weight*tri_loss
            meter.update({'id_loss': id_loss.data,
                            'tri_loss': tri_loss.data})

        else:
            raise RuntimeError('Debug mode {} not found!'.format(args.debug))
    
        total_loss.backward()
        if enable_phase1:
            model.optimizer_phase1.step()
        else:
            model.optimizer_phase2.step()
        
        # 更新进度条显示当前损失
        if hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})
    
    # 关闭进度条
    progress_bar.close()
    
    return meter.get_val(), meter.get_str()

def relabel(select_ids, source_labels, target_labels):
    '''
    Input: source_labels, target_labels
    Output: corresponding select_ids in target modal
    '''
    key_to_value = torch.full((torch.max(source_labels) + 1,), -1, dtype=torch.long).to(source_labels.device)
    key_to_value[source_labels] = target_labels
    
    select_ids = key_to_value[select_ids]
    return select_ids

def hate_nan(loss, condition,logger):
    if torch.isnan(loss):
        if condition:
            logger('no matched labels')
        else:
            logger('nan loss detected')
        return torch.tensor(0.0).to(loss.device)
    else:
        return loss