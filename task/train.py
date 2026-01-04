import torch
import torch.nn as nn
from models import Model
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy
from tqdm import tqdm
import torch.nn.functional as F
import shutil
import os
import sys

def get_diffusion_weight(epoch, phase1_epochs, warmup_epochs=10, max_weight=0.05):
    if epoch < phase1_epochs:
        return 0.0
    
    phase2_epoch = epoch - phase1_epochs
    
    if phase2_epoch < warmup_epochs:
        progress = phase2_epoch / warmup_epochs
        return max_weight * progress
    
    return max_weight

def get_dynamic_threshold(epoch, phase1_epochs, start_val=0.5, end_val=0.8, warm_epochs=30):
    if epoch < phase1_epochs:
        return start_val
    phase2_epoch = epoch - phase1_epochs
    if phase2_epoch < warm_epochs:
        progress = phase2_epoch / warm_epochs
        current_threshold = start_val + (end_val - start_val) * progress
        return current_threshold
    return end_val

def get_terminal_width():
    """Get terminal width for adaptive progress bar"""
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except:
        return 120  # Default fallback


def train(args, model: Model, dataset, epoch, cma: CMA, logger, enable_phase1=False):
    """Main training function with integrated diagnostics (PRUD Enabled)"""
    
    # ==================== Epoch Initialization ====================
    cma.set_epoch(epoch)
    if model.use_diffusion:
        model.diffusion.set_epoch(epoch)
    
    # Phase 2 preparation
    use_prud = (not enable_phase1 and args.use_cycle_consistency and epoch >= args.ccpa_start_epoch)
    match_info = None
    
    if not enable_phase1:
        if 'wsl' in args.debug:
            match_info = _prepare_matching(args, model, dataset, cma)
            n_common = len(match_info['common_rgb'])
            n_specific = len(match_info['specific_rgb'])
            n_remain = len(match_info['remain_rgb'])
            logger(f"\n[Matching] Common: {n_common} | Specific: {n_specific} | Remain: {n_remain}")
    
    if use_prud:
        if epoch == args.ccpa_start_epoch:
            msg = f"✓ PRUD Activated (Prototype-Rectified Distillation)"
            print(msg)
            logger(msg)
        
        if cma.ragm_module is None:
            cma.set_ragm_module(model.ragm)
            
        # Prepare Rectified Prototypes using Diffusion
        cma.prepare_rectified_prototypes(model.diffusion)
    
    # ==================== Training Loop ====================
    model.set_train()
    meter = MultiItemAverageMeter()
    
    rgb_loader, ir_loader = dataset.get_train_loader()
    BATCH_SIZE = args.batch_pidnum * args.pid_numsample
    CHUNK_SIZE = 2 * BATCH_SIZE
    
    phase_name = "Phase 1" if enable_phase1 else "Phase 2"
    
    term_width = get_terminal_width()
    ncols = max(80, min(term_width - 5, 150))
    
    pbar = tqdm(
        enumerate(zip(rgb_loader, ir_loader)),
        total=min(len(rgb_loader), len(ir_loader)),
        desc=f'Epoch {epoch+1:03d} {phase_name}',
        unit='batch',
        ncols=ncols,
        file=sys.stdout,
        dynamic_ncols=True
    )
    
    nan_counter = 0
    
    for iter_idx, ((rgb_imgs, ca_imgs, color_info), (ir_imgs, aug_imgs, ir_info)) in pbar:
        # ==================== Data Preparation ====================
        rgb_imgs, ca_imgs = rgb_imgs.to(model.device), ca_imgs.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        color_imgs = torch.cat((rgb_imgs, ca_imgs), dim=0)
        
        if args.dataset == 'regdb':
            aug_imgs = aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim=0)
        
        rgb_ids = torch.cat((color_info[:, 1], color_info[:, 1])).to(model.device)
        ir_ids = ir_info[:, 1].to(model.device)
        
        if args.dataset == 'regdb':
            ir_ids = torch.cat((ir_ids, ir_ids)).to(model.device)
        
        # ==================== Forward Pass ====================
        gap_features, bn_features = model.model(color_imgs, ir_imgs)
        rgb_features = gap_features[:CHUNK_SIZE]
        ir_features = gap_features[CHUNK_SIZE:]
        
        rgbcls_out, _ = model.classifier1(bn_features)
        ircls_out, _ = model.classifier2(bn_features)
        
        r2r_cls = rgbcls_out[:CHUNK_SIZE]
        i2i_cls = ircls_out[CHUNK_SIZE:]
        r2i_cls = ircls_out[:CHUNK_SIZE]
        i2r_cls = rgbcls_out[CHUNK_SIZE:]
        
        # ==================== Loss Computation ====================
        if enable_phase1:
            total_loss = _compute_phase1_loss(
                model, meter, r2r_cls, i2i_cls,
                rgb_features, ir_features, rgb_ids, ir_ids, args
            )
            optimizer = model.optimizer_phase1
        else:
            # Get Confidence Weights from PRUD
            conf_v, conf_r = None, None
            if use_prud:
                conf_v, conf_r = cma.get_distillation_weights(rgb_ids, ir_ids)
            
            total_loss, nan_counter = _compute_phase2_loss(
                model, meter, bn_features, gap_features, rgb_features, ir_features,
                rgbcls_out, ircls_out, r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                rgb_ids, ir_ids, cma, match_info, epoch, args, nan_counter,
                conf_v, conf_r, use_prud, BATCH_SIZE, CHUNK_SIZE
            )
            optimizer = model.optimizer_phase2
        
        if not torch.isnan(total_loss):
            meter.update({'total': total_loss.data})
        
        # ==================== Backward Pass ====================
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            nan_counter += 1
            if nan_counter < 10:
                logger(f"⚠ WARNING: NaN loss (Count {nan_counter}). Skipping batch.")
                continue
            else:
                error_msg = "❌ ERROR: Too many NaN losses. Breaking."
                print(error_msg)
                logger(error_msg)
                break
        
        optimizer.zero_grad()
        total_loss.backward()

        if model.use_diffusion:
            torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), max_norm=1.0)

        backbone_params = list(model.model.parameters())
        if hasattr(model, 'classifier1'):
            backbone_params += list(model.classifier1.parameters())
        if hasattr(model, 'classifier2'):
            backbone_params += list(model.classifier2.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=5.0)
        optimizer.step()

        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # ==================== Periodic Diagnostics ====================
        if not enable_phase1 and (iter_idx + 1) % 50 == 0:
            _log_diagnostics(
                logger, model, cma, args, epoch, iter_idx,
                grad_norm, optimizer, nan_counter, use_prud
            )
    
    pbar.close()
    
    # ==================== Epoch Summary ====================
    _log_epoch_summary(logger, model, cma, args, meter, epoch, enable_phase1, use_prud)
    
    return meter.get_val(), meter.get_str()


def _log_diagnostics(logger, model, cma, args, epoch, iter_idx, grad_norm, optimizer, nan_counter, use_prud):
    """Centralized diagnostics logging"""
    log_lines = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"🔍 Diagnostic Snapshot | Epoch {epoch+1} | Iteration {iter_idx+1}")
    log_lines.append(f"{'='*80}")
    
    # Training Dynamics
    log_lines.append(f"\n📊 Training Dynamics:")
    log_lines.append(f"  ├─ Gradient Norm: {grad_norm:.4f}")
    log_lines.append(f"  ├─ Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    log_lines.append(f"  └─ NaN Counter: {nan_counter}")
    
    # RAGM Diagnostics
    if hasattr(model, 'ragm'):
        ragm_diag = model.ragm.get_diagnostics()
        log_lines.append(f"\n{ragm_diag}")
    
    # Diffusion Bridge
    if model.use_diffusion and hasattr(model.diffusion, 'get_detailed_diagnostics'):
        diffusion_diag = model.diffusion.get_detailed_diagnostics()
        log_lines.append(f"\n{diffusion_diag}")
    
    # PRUD Status
    if use_prud and hasattr(cma, 'prud'):
        if hasattr(cma.prud, 'get_detailed_diagnostics'):
            prud_diag = cma.prud.get_detailed_diagnostics()
            log_lines.append(f"\n{prud_diag}")
    
    log_lines.append(f"{'='*80}\n")
    logger('\n'.join(log_lines))


def _log_epoch_summary(logger, model, cma, args, meter, epoch, enable_phase1, use_prud):
    """Epoch-level summary"""
    avg_losses = meter.get_avg()
    lr = model.optimizer_phase1.param_groups[0]['lr'] if enable_phase1 else model.optimizer_phase2.param_groups[0]['lr']
    phase_tag = "P1" if enable_phase1 else "P2"
    
    log_lines = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"📌 EPOCH {epoch+1} SUMMARY ({phase_tag})")
    log_lines.append(f"{'='*80}")
    log_lines.append(f"Learning Rate: {lr:.2e}")
    log_lines.append(f"Total Loss: {avg_losses.get('total', 0):.4f}")
    
    # Loss Breakdown
    loss_details = []
    possible_keys = ['p1_id', 'p1_tri', 'p2_id', 'tri', 'diff', 'weight_reg', 'prud', 'cross', 'weak', 'cma']
    for key in possible_keys:
        if key in avg_losses:
            loss_details.append(f"  - {key.upper()}: {avg_losses[key]:.4f}")
    
    if loss_details:
        log_lines.append(f"\n📊 Loss Components:")
        log_lines.extend(loss_details)
    
    # Memory Bank Summary
    if not enable_phase1 and model.use_diffusion and hasattr(model.diffusion, 'memory_bank'):
        mem_stats = model.diffusion.memory_bank.get_statistics()
        curr_thresh = get_dynamic_threshold(epoch, args.stage1_epoch)
        log_lines.append(f"\n💾 Memory Bank Summary (Threshold: {curr_thresh:.2f}):")
        log_lines.append(f"  ├─ Occupancy: Vis {mem_stats['occupied_classes_v']}/{args.num_classes} | IR {mem_stats['occupied_classes_r']}/{args.num_classes}")
        log_lines.append(f"  ├─ Avg Quality: Vis {mem_stats['avg_quality_v']:.3f} | IR {mem_stats['avg_quality_r']:.3f}")
    
    log_lines.append(f"{'='*80}\n")
    logger('\n'.join(log_lines))


# ==================== Helper Functions ====================

def _prepare_matching(args, model, dataset, cma):
    """Optimized matching preparation using CMA API"""
    cma.extract(args, model, dataset)
    r2i_pair, i2r_pair = cma.get_label()
    
    common_dict, specific_dict, remain_dict = {}, {}, {}
    for r, i in r2i_pair.items():
        if i in i2r_pair and i2r_pair[i] == r:
            common_dict[r] = i
        elif r not in i2r_pair.values() and i not in i2r_pair:
            specific_dict[r] = i
        else:
            remain_dict[r] = i
            
    for i, r in i2r_pair.items():
        if (r, i) in common_dict.items(): continue
        if r not in r2i_pair.values() and i not in r2i_pair:
            specific_dict[r] = i
        else:
            remain_dict[r] = i
            
    device = model.device
    num_classes = args.num_classes
    
    common_rm = torch.zeros((num_classes, num_classes), device=device)
    specific_rm = torch.zeros((num_classes, num_classes), device=device)
    remain_rm = torch.zeros((num_classes, num_classes), device=device)
    
    for r, i in common_dict.items(): common_rm[r, i] = 1
    for r, i in specific_dict.items(): specific_rm[r, i] = 1
    for r, i in remain_dict.items(): remain_rm[r, i] = 1
    specific_rm += common_rm
    
    return {
        'common_rm': common_rm,
        'specific_rm': specific_rm,
        'remain_rm': remain_rm,
        'common_rgb': torch.tensor(list(common_dict.keys()), device=device),
        'common_ir': torch.tensor(list(common_dict.values()), device=device),
        'specific_rgb': torch.tensor(list(specific_dict.keys()), device=device),
        'remain_rgb': torch.tensor(list(remain_dict.keys()), device=device),
        'remain_ir': torch.tensor(list(remain_dict.values()), device=device)
    }


def _compute_phase1_loss(model, meter, r2r_cls, i2i_cls, rgb_feat, ir_feat, rgb_ids, ir_ids, args):
    """Standard supervised loss"""
    loss_id = model.pid_criterion(r2r_cls, rgb_ids) + model.pid_criterion(i2i_cls, ir_ids)
    loss_tri = (model.tri_criterion(rgb_feat, rgb_ids) + model.tri_criterion(ir_feat, ir_ids)) * args.tri_weight
    meter.update({'p1_id': loss_id.data, 'p1_tri': loss_tri.data})
    return loss_id + loss_tri


def _compute_phase2_loss(model, meter, bn_features, gap_features, rgb_features, ir_features,
                        rgbcls_out, ircls_out, r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                        rgb_ids, ir_ids, cma, match_info, epoch, args, nan_counter,
                        conf_v, conf_r, use_prud, BATCH_SIZE, CHUNK_SIZE):
    
    if args.debug == 'baseline':
        return _compute_phase1_loss(model, meter, r2r_cls, i2i_cls, rgb_features, ir_features, rgb_ids, ir_ids, args), nan_counter
    
    # Base ID loss
    dtd_feats = bn_features
    dtd_r2r = model.classifier1(dtd_feats)[0][:CHUNK_SIZE]
    dtd_i2i = model.classifier2(dtd_feats)[0][CHUNK_SIZE:]
    
    loss_id = model.pid_criterion(dtd_r2r, rgb_ids) + model.pid_criterion(dtd_i2i, ir_ids)
    total_loss = loss_id
    meter.update({'p2_id': loss_id.data})
    
    warmup_epochs = 10
    is_warmup = (epoch < args.stage1_epoch + warmup_epochs)
    rel_v2r = torch.ones(BATCH_SIZE, device=model.device)
    
    # ==================== Diffusion Bridge ====================
    if model.use_diffusion:
        f_v_input = bn_features[:BATCH_SIZE] 
        f_r_input = bn_features[CHUNK_SIZE : CHUNK_SIZE + BATCH_SIZE]
        diff_loss_dict, f_r_fake, f_v_fake, unc_v, unc_r = model.diffusion(
            f_v=f_v_input,  
            f_r=f_r_input, 
            labels_v=rgb_ids[:BATCH_SIZE],
            labels_r=ir_ids[:BATCH_SIZE],
            mode='train'
        )
        
        diff_loss_total = diff_loss_dict['total'] / 4.0
        current_diff_weight = get_diffusion_weight(epoch, args.stage1_epoch, warmup_epochs, args.diffusion_weight)
        total_loss += current_diff_weight * diff_loss_total
        meter.update({'diff': diff_loss_total.data})
        if hasattr(model.diffusion, 'denoiser_vr'):
            weight_reg, _ = model.diffusion.denoiser_vr.compute_weight_reg()
            total_loss += weight_reg
            meter.update({'weight_reg': weight_reg.data})
        
        # Memory bank update
        if epoch >= args.stage1_epoch:
            with torch.no_grad():
                recon_loss_v = F.mse_loss(f_v_fake, f_v_input, reduction='none').mean(dim=1)
                recon_loss_r = F.mse_loss(f_r_fake, f_r_input, reduction='none').mean(dim=1)
                quality_v = torch.clamp(1.0 / (1.0 + recon_loss_v), 0.0, 1.0)
                quality_r = torch.clamp(1.0 / (1.0 + recon_loss_r), 0.0, 1.0)
                
                dynamic_threshold = get_dynamic_threshold(epoch, args.stage1_epoch)
                
                model.diffusion.memory_bank.update(
                    keys=f_v_input.detach(), values=f_r_fake.detach(),
                    labels=rgb_ids[:BATCH_SIZE], quality_scores=quality_v,
                    modality='v', threshold=dynamic_threshold
                )
                model.diffusion.memory_bank.update(
                    keys=f_r_input.detach(), values=f_v_fake.detach(),
                    labels=ir_ids[:BATCH_SIZE], quality_scores=quality_r,
                    modality='r', threshold=dynamic_threshold
                )
        
        # RAGM reliability prediction
        # ✅ FIX: Clone prototypes to avoid in-place modification error during backward
        _, rel_v2r_raw = model.ragm(
            f_r_fake.detach(), 
            cma.get_prototypes('r').clone().detach(), 
            uncertainty=unc_v
        )
        if not torch.isnan(rel_v2r_raw).any():
            rel_v2r = rel_v2r_raw

    # ==================== PRUD Loss (Replaces CCPA) ====================
    if use_prud and not is_warmup and conf_v is not None:
        ce_none = nn.CrossEntropyLoss(reduction='none')
        loss_prud_v = (ce_none(r2i_cls[:BATCH_SIZE], rgb_ids[:BATCH_SIZE]) * conf_v[:BATCH_SIZE]).mean()
        loss_prud_r = (ce_none(i2r_cls[:BATCH_SIZE], ir_ids[:BATCH_SIZE]) * conf_r[:BATCH_SIZE]).mean()
        loss_prud = (loss_prud_v + loss_prud_r) / 2.0
        total_loss += args.ccpa_weight * loss_prud
        meter.update({'prud': loss_prud.data})
    
    # ==================== Triplet Loss ====================
    mask_rgb = torch.isin(rgb_ids[:BATCH_SIZE], match_info['common_rgb'])
    mask_ir = torch.isin(ir_ids, match_info['common_ir'])
    
    if mask_rgb.any() and mask_ir.any():
        tri_loss = _compute_weighted_triplet(
            model, rgb_features[:BATCH_SIZE], ir_features,
            rgb_ids[:BATCH_SIZE], ir_ids,
            mask_rgb, mask_ir, match_info['common_rm'],
            args.tri_weight, rel_v2r
        )
        total_loss += tri_loss
        meter.update({'tri': tri_loss.data})
    
    # ==================== CMA Memory Alignment ====================
    # Always update prototypes (Now handled by cma.update_memory for PRUD source)
    cma.update_prototypes(bn_features[:CHUNK_SIZE], rgb_ids, 'v')
    cma.update_prototypes(bn_features[CHUNK_SIZE:], ir_ids, 'r')   
    
    if not is_warmup:
        cma.update_memory(bn_features[:CHUNK_SIZE], bn_features[CHUNK_SIZE:], rgb_ids, ir_ids)
        cma_loss = _compute_cma_loss(
            model, cma, dtd_r2r[:BATCH_SIZE], dtd_i2i[:BATCH_SIZE],
            r2i_cls[:BATCH_SIZE], i2r_cls[:BATCH_SIZE],
            rgb_ids[:BATCH_SIZE], ir_ids[:BATCH_SIZE],
            mask_rgb, mask_ir[:BATCH_SIZE],
            match_info['common_rm']
        )
        if cma_loss is not None:
            total_loss += cma_loss
            meter.update({'cma': cma_loss.data})
    
    # ==================== Weak Supervision ====================
    if epoch >= args.stage1_epoch + 1:
        weak_loss = _compute_weak_loss(
            model, bn_features, rgb_ids[:BATCH_SIZE],
            match_info['remain_rgb'], match_info['remain_rm'],
            args.weak_weight, rel_v2r, BATCH_SIZE
        )
        if weak_loss is not None:
            total_loss += weak_loss
            meter.update({'weak': weak_loss.data})
            
    return total_loss, nan_counter


def _compute_weighted_triplet(model, rgb_feats, ir_feats, rgb_ids, ir_ids,
                              mask_rgb, mask_ir, common_rm, weight, reliability):
    """Triplet loss with reliability weighting"""
    sel_rgb_f = rgb_feats[mask_rgb]
    sel_ir_f = ir_feats[mask_ir]
    sel_rgb_id = rgb_ids[mask_rgb]
    sel_ir_id = ir_ids[mask_ir]
    
    loss = weight * (
        model.tri_criterion(sel_rgb_f, sel_rgb_id) +
        model.tri_criterion(sel_ir_f, sel_ir_id)
    )
    return loss


def _compute_cma_loss(model, cma, dtd_r2r, dtd_i2i, r2i, i2r, rgb_ids, ir_ids,
                     mask_rgb, mask_ir, common_rm):
    """Memory alignment loss"""
    if not mask_rgb.any() or not mask_ir.any():
        return None
    
    p_r2i = F.softmax(r2i, dim=1)
    p_i2r = F.softmax(i2r, dim=1)
    
    ent_r2i = infoEntropy(p_r2i)
    ent_i2r = infoEntropy(p_i2r)
    
    w_r2i = ent_r2i / (ent_r2i + ent_i2r + 1e-8)
    w_i2r = ent_i2r / (ent_r2i + ent_i2r + 1e-8)
    
    criterion = nn.MSELoss()
    loss = 0.0
    
    if mask_rgb.any():
        sel_rgb_ids = rgb_ids[mask_rgb]
        mapped_ir_idx = torch.nonzero(common_rm[sel_rgb_ids])[:, -1]
        mem_r = cma.ir_memory[mapped_ir_idx].detach()
        logits_mem_r = model.classifier2(mem_r)[0]
        loss += w_i2r * criterion(dtd_r2r[mask_rgb], logits_mem_r)
    
    if mask_ir.any():
        sel_ir_ids = ir_ids[mask_ir]
        mapped_rgb_idx = torch.nonzero(common_rm.T[sel_ir_ids])[:, -1]
        mem_v = cma.vis_memory[mapped_rgb_idx].detach()
        logits_mem_v = model.classifier1(mem_v)[0]
        loss += w_r2i * criterion(dtd_i2i[mask_ir], logits_mem_v)
    
    return loss


def _compute_weak_loss(model, bn_features, rgb_ids, remain_rgb, remain_rm,
                      weight, reliability, BATCH_SIZE):
    """Weak supervision for unmatched samples"""
    mask = torch.isin(rgb_ids, remain_rgb)
    if not mask.any():
        return None
    
    r2c_cls = model.classifier3(bn_features)[0][:BATCH_SIZE]
    sel_pred = r2c_cls[mask]
    sel_target = remain_rm[rgb_ids[mask]]
    
    if reliability is not None:
        loss_raw = model.weak_criterion(sel_pred, sel_target, reduction='none')
        loss = (loss_raw * reliability[mask]).mean()
    else:
        loss = model.weak_criterion(sel_pred, sel_target)
    
    return weight * loss