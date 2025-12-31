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
    """Main training function with integrated diagnostics"""
    
    # ==================== Epoch Initialization ====================
    cma.set_epoch(epoch)
    if model.use_diffusion:
        model.diffusion.set_epoch(epoch)
    
    # Phase 2 preparation
    use_ccpa = (not enable_phase1 and args.use_cycle_consistency and epoch >= args.ccpa_start_epoch)
    match_info = None
    
    if not enable_phase1:
        if 'wsl' in args.debug:
            match_info = _prepare_matching(args, model, dataset, cma)
            n_common = len(match_info['common_rgb'])
            n_specific = len(match_info['specific_rgb'])
            n_remain = len(match_info['remain_rgb'])
            
            # Log to file only
            logger(f"\n[Matching] Common: {n_common} | Specific: {n_specific} | Remain: {n_remain}")
    
    if use_ccpa:
        if epoch == args.ccpa_start_epoch:
            msg = f"✓ CCPA Activated (Mode: {args.ccpa_threshold_mode})"
            print(msg)  # Terminal
            logger(msg)  # File
        
        if cma.ragm_module is None:
            cma.set_ragm_module(model.ragm)
        cma.prepare_cycle_matching(model.diffusion)
    
    # ==================== Training Loop ====================
    model.set_train()
    meter = MultiItemAverageMeter()
    
    rgb_loader, ir_loader = dataset.get_train_loader()
    BATCH_SIZE = args.batch_pidnum * args.pid_numsample
    CHUNK_SIZE = 2 * BATCH_SIZE
    
    phase_name = "Phase 1" if enable_phase1 else "Phase 2"
    
    # Adaptive progress bar width (FIX: actually get terminal width)
    term_width = get_terminal_width()
    # Reserve space for description and stats
    ncols = max(80, min(term_width - 5, 150))
    
    # CRITICAL FIX: Use file=sys.stdout to ensure output to terminal
    pbar = tqdm(
        enumerate(zip(rgb_loader, ir_loader)),
        total=min(len(rgb_loader), len(ir_loader)),
        desc=f'Epoch {epoch+1:03d} {phase_name}',
        unit='batch',
        ncols=ncols,
        file=sys.stdout,  # Explicitly use stdout
        dynamic_ncols=True  # Allow dynamic width adjustment
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
            # CCPA reliability masks
            rel_mask_v, rel_mask_r = None, None
            if use_ccpa:
                rel_mask_v, rel_mask_r, _, _ = cma.compute_cycle_consistent_matching(
                    rgb_features, ir_features, rgb_ids, ir_ids
                )
            
            total_loss, nan_counter = _compute_phase2_loss(
                model, meter, bn_features, gap_features, rgb_features, ir_features,
                rgbcls_out, ircls_out, r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                rgb_ids, ir_ids, cma, match_info, epoch, args, nan_counter,
                rel_mask_v, rel_mask_r, use_ccpa, BATCH_SIZE, CHUNK_SIZE
            )
            optimizer = model.optimizer_phase2
        
        if not torch.isnan(total_loss):
            meter.update({'total': total_loss.data})
        
        # ==================== Backward Pass ====================
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            nan_counter += 1
            if nan_counter < 10:
                warning_msg = f"⚠ WARNING: NaN loss (Count {nan_counter}). Skipping batch."
                logger(warning_msg)  # File only
                continue
            else:
                error_msg = "❌ ERROR: Too many NaN losses. Breaking."
                print(error_msg)
                logger(error_msg)
                break
        
        optimizer.zero_grad()
        total_loss.backward()

        # 分模块梯度裁剪（扩散模块使用更严格的限制）
        if model.use_diffusion:
            torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), max_norm=1.0)

        # 其他模块使用标准裁剪
        backbone_params = list(model.model.parameters())
        if hasattr(model, 'classifier1'):
            backbone_params += list(model.classifier1.parameters())
        if hasattr(model, 'classifier2'):
            backbone_params += list(model.classifier2.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=5.0)
        optimizer.step()

        # Update progress bar (only loss)
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # ==================== Periodic Diagnostics (File Only) ====================
        if not enable_phase1 and (iter_idx + 1) % 50 == 0:
            _log_diagnostics(
                logger, model, cma, args, epoch, iter_idx,
                grad_norm, optimizer, nan_counter, use_ccpa
            )
    
    pbar.close()
    
    # ==================== Epoch Summary ====================
    _log_epoch_summary(logger, model, cma, args, meter, epoch, enable_phase1, use_ccpa)
    
    return meter.get_val(), meter.get_str()


def _log_diagnostics(logger, model, cma, args, epoch, iter_idx, grad_norm, optimizer, nan_counter, use_ccpa):
    """
    Centralized diagnostics logging (FILE ONLY - no terminal output).
    """
    log_lines = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"🔍 Diagnostic Snapshot | Epoch {epoch+1} | Iteration {iter_idx+1}")
    log_lines.append(f"{'='*80}")
    
    # ==================== Training Dynamics ====================
    log_lines.append(f"\n📊 Training Dynamics:")
    log_lines.append(f"  ├─ Gradient Norm: {grad_norm:.4f}")
    log_lines.append(f"  ├─ Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    log_lines.append(f"  └─ NaN Counter: {nan_counter}")
    
    # ==================== Loss Weights ====================
    current_diff_weight = get_diffusion_weight(epoch, args.stage1_epoch, 10, args.diffusion_weight)
    current_threshold = get_dynamic_threshold(epoch, args.stage1_epoch)
    
    log_lines.append(f"\n⚖️ Loss Weights & Schedulers:")
    log_lines.append(f"  ├─ Diffusion Weight: {current_diff_weight:.4f} (Target: {args.diffusion_weight:.4f})")
    log_lines.append(f"  ├─ Memory Threshold: {current_threshold:.3f}")
    
    if use_ccpa:
        log_lines.append(f"  └─ CCPA Weight: {args.ccpa_weight:.4f}")
    
    # ==================== RAGM Diagnostics ====================
    if hasattr(model, 'ragm'):
        ragm_diag = model.ragm.get_diagnostics()
        log_lines.append(f"\n{ragm_diag}")
    
    # ==================== Prototype Guidance ====================
    if model.use_diffusion and hasattr(model.diffusion, 'proto_guidance'):
        proto_diag = model.diffusion.proto_guidance.get_diagnostics()
        log_lines.append(f"\n{proto_diag}")
    
    # ==================== Diffusion Bridge ====================
    if model.use_diffusion and hasattr(model.diffusion, 'get_detailed_diagnostics'):
        diffusion_diag = model.diffusion.get_detailed_diagnostics()
        log_lines.append(f"\n{diffusion_diag}")
    
    # ==================== Memory Bank ====================
    if model.use_diffusion and hasattr(model.diffusion, 'memory_bank'):
        mem_diag = model.diffusion.memory_bank.get_detailed_diagnostics(current_threshold)
        log_lines.append(f"\n{mem_diag}")
    
    # ==================== CCPA Status ====================
    if use_ccpa and hasattr(cma, 'ccpa'):
        ccpa_stats = cma.ccpa.get_pseudo_quality_info()
        log_lines.append(f"\n🔄 CCPA (Cycle-Consistent Pseudo Assignment):")
        log_lines.append(f"  ├─ Status: {'✓ Active' if ccpa_stats['pseudo_ready'] else '⏳ Preparing'}")
        log_lines.append(f"  ├─ Proto Quality (V→R): {ccpa_stats['quality_v2r']:.4f}")
        log_lines.append(f"  ├─ Proto Quality (R→V): {ccpa_stats['quality_r2v']:.4f}")
        log_lines.append(f"  └─ Fallback Mode: {ccpa_stats['use_fallback']}")
    
    log_lines.append(f"{'='*80}\n")
    
    # Write to logger (FILE ONLY)
    logger('\n'.join(log_lines))


def _log_epoch_summary(logger, model, cma, args, meter, epoch, enable_phase1, use_ccpa):
    """
    Epoch-level summary (FILE ONLY).
    """
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
    possible_keys = ['p1_id', 'p1_tri', 'p2_id', 'tri', 'diff', 'weight_reg', 'ccpa', 'cross', 'weak', 'cma']
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
        log_lines.append(f"  ├─ Epoch Updates: Vis +{mem_stats['updates_v']} | IR +{mem_stats['updates_r']}")
        log_lines.append(f"  └─ Rejected: Vis {mem_stats['rejected_v']} | IR {mem_stats['rejected_r']}")
    
    # CCPA Summary
    if not enable_phase1 and use_ccpa and hasattr(cma, 'ccpa'):
        ccpa_stats = cma.ccpa.get_pseudo_quality_info()
        log_lines.append(f"\n🔄 CCPA Summary:")
        log_lines.append(f"  ├─ Proto Quality: V→R {ccpa_stats['quality_v2r']:.3f} | R→V {ccpa_stats['quality_r2v']:.3f}")
        log_lines.append(f"  └─ State: {'Ready' if ccpa_stats['pseudo_ready'] else 'Not Ready'} | Fallback: {ccpa_stats['use_fallback']}")
    
    log_lines.append(f"{'='*80}\n")
    
    logger('\n'.join(log_lines))

# ==================== Helper Functions ====================

def _prepare_matching(args, model, dataset, cma):
    """Optimized matching preparation using CMA API"""
    cma.extract(args, model, dataset)
    r2i_pair, i2r_pair = cma.get_label()
    
    # Compute association masks
    common_dict, specific_dict, remain_dict = {}, {}, {}
    
    for r, i in r2i_pair.items():
        if i in i2r_pair and i2r_pair[i] == r:
            common_dict[r] = i
        elif r not in i2r_pair.values() and i not in i2r_pair:
            specific_dict[r] = i
        else:
            remain_dict[r] = i
    
    for i, r in i2r_pair.items():
        if (r, i) in common_dict.items():
            continue
        if r not in r2i_pair.values() and i not in r2i_pair:
            specific_dict[r] = i
        else:
            remain_dict[r] = i
    
    # Build masks
    device = model.device
    num_classes = args.num_classes
    
    common_rm = torch.zeros((num_classes, num_classes), device=device)
    specific_rm = torch.zeros((num_classes, num_classes), device=device)
    remain_rm = torch.zeros((num_classes, num_classes), device=device)
    
    for r, i in common_dict.items():
        common_rm[r, i] = 1
    
    for r, i in specific_dict.items():
        specific_rm[r, i] = 1
    
    for r, i in remain_dict.items():
        remain_rm[r, i] = 1
    
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
                        rel_mask_v, rel_mask_r, use_ccpa, BATCH_SIZE, CHUNK_SIZE):
    
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
        diff_loss_dict, f_r_fake, f_v_fake, unc_v, unc_r = model.diffusion(
            f_v=rgb_features[:BATCH_SIZE],
            f_r=ir_features[:BATCH_SIZE],
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
                recon_loss_v = F.mse_loss(f_v_fake, rgb_features[:BATCH_SIZE], reduction='none').mean(dim=1)
                recon_loss_r = F.mse_loss(f_r_fake, ir_features[:BATCH_SIZE], reduction='none').mean(dim=1)
                
                quality_v = 1.0 / (1.0 + recon_loss_v)
                quality_r = 1.0 / (1.0 + recon_loss_r)
                
                quality_v = torch.clamp(quality_v, 0.0, 1.0)
                quality_r = torch.clamp(quality_r, 0.0, 1.0)
                
                dynamic_threshold = get_dynamic_threshold(epoch, args.stage1_epoch)
                
                model.diffusion.memory_bank.update(
                    keys=rgb_features[:BATCH_SIZE].detach(),
                    values=f_r_fake.detach(),
                    labels=rgb_ids[:BATCH_SIZE],
                    quality_scores=quality_v,
                    modality='v',
                    threshold=dynamic_threshold
                )
                
                model.diffusion.memory_bank.update(
                    keys=ir_features[:BATCH_SIZE].detach(),
                    values=f_v_fake.detach(),
                    labels=ir_ids[:BATCH_SIZE],
                    quality_scores=quality_r,
                    modality='r',
                    threshold=dynamic_threshold
                )
        
        # RAGM reliability prediction
        _, rel_v2r_raw = model.ragm(f_r_fake.detach(), cma.get_prototypes('r'), uncertainty=unc_v)
        _, _ = model.ragm(f_v_fake.detach(), cma.get_prototypes('v'), uncertainty=unc_r)
        
        if not torch.isnan(rel_v2r_raw).any():
            rel_v2r = rel_v2r_raw
        
        # Fusion with CCPA reliability
        if rel_mask_v is not None and use_ccpa:
            alpha = 0.5
            rel_v2r = alpha * rel_v2r + (1 - alpha) * rel_mask_v[:BATCH_SIZE]
            rel_v2r = 0.05 + 0.95 * torch.sigmoid(10 * (rel_v2r - 0.5))
    
    # ==================== CCPA Loss ====================
    if use_ccpa and not is_warmup and rel_mask_v is not None:
        ce_none = nn.CrossEntropyLoss(reduction='none')
        loss_ccpa_v = (ce_none(r2i_cls[:BATCH_SIZE], ir_ids[:BATCH_SIZE]) * rel_mask_v[:BATCH_SIZE]).mean()
        loss_ccpa = loss_ccpa_v
        
        total_loss += args.ccpa_weight * loss_ccpa
        meter.update({'ccpa': loss_ccpa.data})
    
    # ==================== Triplet Loss (Reliability-Weighted) ====================
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
    if not is_warmup:
        cma.update_memory(bn_features[:CHUNK_SIZE], bn_features[CHUNK_SIZE:], rgb_ids, ir_ids)
        cma.update_prototypes(rgb_features[:CHUNK_SIZE], rgb_ids, 'v')
        cma.update_prototypes(ir_features, ir_ids, 'r')
        
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
    
    # ==================== Weak Supervision Loss ====================
    if epoch >= args.stage1_epoch + 1:
        weak_loss = _compute_weak_loss(
            model, bn_features, rgb_ids[:BATCH_SIZE],
            match_info['remain_rgb'], match_info['remain_rm'],
            args.weak_weight, rel_v2r, BATCH_SIZE
        )
        
        if weak_loss is not None:
            total_loss += weak_loss
            meter.update({'weak': weak_loss.data})
    
    # ==================== Cross-Modal Classification ====================
    if model.enable_cls3:
        idx_spec = torch.isin(rgb_ids[:BATCH_SIZE], match_info['specific_rgb'])
        idx_spec = idx_spec & (~mask_rgb)
        
        if idx_spec.any():
            r2c_cls = model.classifier3(bn_features)[0][:BATCH_SIZE]
            target_dist = match_info['specific_rm'][rgb_ids[:BATCH_SIZE][idx_spec]]
            loss_cross = model.pid_criterion(r2c_cls[idx_spec], target_dist)
            total_loss += loss_cross
            meter.update({'cross': loss_cross.data})
    
    return total_loss, nan_counter


def _compute_weighted_triplet(model, rgb_feats, ir_feats, rgb_ids, ir_ids,
                              mask_rgb, mask_ir, common_rm, weight, reliability):
    """Triplet loss with reliability weighting"""
    sel_rgb_f = rgb_feats[mask_rgb]
    sel_ir_f = ir_feats[mask_ir]
    sel_rgb_id = rgb_ids[mask_rgb]
    sel_ir_id = ir_ids[mask_ir]
    
    trans_rgb_lbl = torch.nonzero(common_rm[sel_rgb_id])[:, -1]
    trans_ir_lbl = torch.nonzero(common_rm.T[sel_ir_id])[:, -1]
    
    match_rgb_f = torch.cat((sel_rgb_f, ir_feats), dim=0)
    match_ir_f = torch.cat((rgb_feats, sel_ir_f), dim=0)
    
    match_rgb_l = torch.cat((trans_rgb_lbl, ir_ids), dim=0)
    match_ir_l = torch.cat((rgb_ids, trans_ir_lbl), dim=0)
    
    w_rgb = None
    if reliability is not None:
        sel_rel = reliability[mask_rgb]
        w_rgb = torch.cat((sel_rel, torch.ones_like(ir_ids, dtype=torch.float32)))
    
    loss = weight * (
        model.tri_criterion(match_rgb_f, match_rgb_l, weights=w_rgb) +
        model.tri_criterion(match_ir_f, match_ir_l)
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
