"""
训练模块 - 集成特征扩散桥
"""
import torch
from models import Model
from collections import OrderedDict
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy
from tqdm import tqdm


def train(args, model: Model, dataset, *arg):
    """主训练函数"""
    epoch, cma, logger, enable_phase1 = arg[0], arg[1], arg[2], arg[3]
    
    # Phase 2 需要提取特征并进行标签匹配
    if not enable_phase1:
        match_info = _prepare_matching(args, model, dataset, cma, epoch)
    else:
        match_info = None
    
    # 设置训练模式
    model.set_train()
    meter = MultiItemAverageMeter()
    
    # 数据加载器
    rgb_loader, ir_loader = dataset.get_train_loader()
    bt = args.batch_pidnum * args.pid_numsample
    
    # 进度条
    phase = "Phase1" if enable_phase1 else "Phase2"
    pbar = tqdm(
        zip(rgb_loader, ir_loader),
        total=min(len(rgb_loader), len(ir_loader)),
        desc=f'Epoch {epoch} {phase}',
        unit='batch'
    )
    
    nan_counter = 0
    
    for (rgb_imgs, ca_imgs, color_info), (ir_imgs, aug_imgs, ir_info) in pbar:
        # ========== 数据准备 ==========
        rgb_imgs, ca_imgs = rgb_imgs.to(model.device), ca_imgs.to(model.device)
        ir_imgs = ir_imgs.to(model.device)
        
        # 拼接数据
        color_imgs = torch.cat((rgb_imgs, ca_imgs), dim=0)
        if args.dataset == 'regdb':
            aug_imgs = aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim=0)
        
        # 标签
        rgb_ids = torch.cat((color_info[:,1], color_info[:,1])).to(model.device)
        ir_ids = ir_info[:,1].to(model.device)
        if args.dataset == 'regdb':
            ir_ids = torch.cat((ir_ids, ir_ids)).to(model.device)
        
        # ========== 前向传播 ==========
        gap_features, bn_features = model.model(color_imgs, ir_imgs)
        rgb_features, ir_features = gap_features[:2*bt], gap_features[2*bt:]
        
        # 分类器输出
        rgbcls_out, _ = model.classifier1(bn_features)
        ircls_out, _ = model.classifier2(bn_features)
        
        r2r_cls, i2i_cls = rgbcls_out[:2*bt], ircls_out[2*bt:]
        r2i_cls, i2r_cls = ircls_out[:2*bt], rgbcls_out[2*bt:]
        
        # ========== 计算损失 ==========
        if enable_phase1:
            # Phase 1: 简单监督学习
            total_loss = _compute_phase1_loss(
                model, meter, r2r_cls, i2i_cls, rgb_features, ir_features, rgb_ids, ir_ids, args
            )
            optimizer = model.optimizer_phase1
        else:
            # Phase 2: WSL + 扩散模块
            total_loss, nan_counter = _compute_phase2_loss(
                model, meter, bn_features, gap_features, rgb_features, ir_features,
                rgbcls_out, ircls_out, r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                rgb_ids, ir_ids, cma, match_info, epoch, args, nan_counter
            )
            optimizer = model.optimizer_phase2
        
        # ========== 反向传播 ==========
        optimizer.zero_grad()
        total_loss.backward()
        # ========== 新增：梯度裁剪 ==========
        torch.nn.utils.clip_grad_norm_(
            model.model.parameters(), max_norm=5.0
        )
        if model.use_diffusion:
            torch.nn.utils.clip_grad_norm_(
                model.diffusion_bridge.parameters(), max_norm=5.0
            )
        # ===================================
        optimizer.step()
        
        # 更新进度条
        postfix = {'loss': f'{total_loss.item():.4f}'}
        if nan_counter > 0:
            postfix['nan'] = nan_counter
        pbar.set_postfix(postfix)
    
    pbar.close()
    return meter.get_val(), meter.get_str()


def _prepare_matching(args, model, dataset, cma, epoch):
    """准备跨模态匹配信息"""
    if 'wsl' not in args.debug:
        return None
    
    # 提取特征并生成伪标签
    cma.extract(args, model, dataset)
    r2i_pair_dict, i2r_pair_dict = cma.get_label(epoch)
    
    # 构建匹配字典
    common_dict, specific_dict, remain_dict = {}, {}, {}
    
    for r, i in r2i_pair_dict.items():
        if i in i2r_pair_dict and i2r_pair_dict[i] == r:
            common_dict[r] = i  # 双向匹配
        elif r not in i2r_pair_dict.values() and i not in i2r_pair_dict:
            specific_dict[r] = i  # RGB特定
        else:
            remain_dict[r] = i  # 不确定
    
    for i, r in i2r_pair_dict.items():
        if (r, i) in common_dict.items():
            continue
        elif r not in r2i_pair_dict.values() and i not in r2i_pair_dict:
            specific_dict[r] = i  # IR特定
        else:
            remain_dict[r] = i
    
    # 构建关系矩阵
    device = model.device
    num_classes = args.num_classes
    common_rm = torch.zeros((num_classes, num_classes)).to(device)
    specific_rm = torch.zeros((num_classes, num_classes)).to(device)
    remain_rm = torch.zeros((num_classes, num_classes)).to(device)
    
    for r, i in common_dict.items():
        common_rm[r, i] = 1
    for r, i in specific_dict.items():
        specific_rm[r, i] = 1
    for r, i in remain_dict.items():
        remain_rm[r, i] = 1
    
    specific_rm += common_rm
    
    # 启用第三个分类器
    if not model.enable_cls3:
        model.enable_cls3 = True
    
    return {
        'common_rm': common_rm,
        'specific_rm': specific_rm,
        'remain_rm': remain_rm,
        'common_matched_rgb': torch.tensor(list(common_dict.keys())).to(device),
        'common_matched_ir': torch.tensor(list(common_dict.values())).to(device),
        'specific_matched_rgb': torch.tensor(list(specific_dict.keys())).to(device),
        'remain_matched_rgb': torch.tensor(list(remain_dict.keys())).to(device),
        'remain_matched_ir': torch.tensor(list(remain_dict.values())).to(device)
    }


def _compute_phase1_loss(model, meter, r2r_cls, i2i_cls, rgb_features, ir_features, rgb_ids, ir_ids, args):
    """Phase 1 损失：简单分类 + 三元组"""
    r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
    i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
    r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
    i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
    
    total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
    
    meter.update({
        'r2r_id': r2r_id_loss.data,
        'i2i_id': i2i_id_loss.data,
        'r2r_tri': r2r_tri_loss.data,
        'i2i_tri': i2i_tri_loss.data
    })
    
    return total_loss


def _compute_phase2_loss(model, meter, bn_features, gap_features, rgb_features, ir_features,
                         rgbcls_out, ircls_out, r2r_cls, i2i_cls, r2i_cls, i2r_cls,
                         rgb_ids, ir_ids, cma, match_info, epoch, args, nan_counter):
    """Phase 2 损失：WSL + 扩散桥"""
    
    if args.debug == 'baseline':
        return _baseline_loss(model, meter, r2r_cls, i2i_cls, rgb_features, ir_features, rgb_ids, ir_ids, args), nan_counter
    
    if args.debug == 'sl':
        return _supervised_loss(model, meter, rgbcls_out, gap_features, args), nan_counter
    
    if args.debug != 'wsl':
        raise ValueError(f"Unknown debug mode: {args.debug}")
    
    # ========== WSL 基础损失 ==========
    dtd_features = bn_features.detach()
    dtd_r2r_cls = model.classifier1(dtd_features)[0][:2*args.batch_pidnum*args.pid_numsample]
    dtd_i2i_cls = model.classifier2(dtd_features)[0][2*args.batch_pidnum*args.pid_numsample:]
    
    r2r_id_loss = model.pid_criterion(dtd_r2r_cls, rgb_ids)
    i2i_id_loss = model.pid_criterion(dtd_i2i_cls, ir_ids)
    total_loss = r2r_id_loss + i2i_id_loss
    meter.update({'r2r_id': r2r_id_loss.data, 'i2i_id': i2i_id_loss.data})
    
    # ========== 扩散桥生成红外特征 ==========
    f_r_fake = None
    if model.use_diffusion:
        W_r = model.classifier2.classifier.weight  # 红外分类器权重
        diffusion_loss_dict, f_r_fake = model.diffusion_bridge(rgb_features, W_r, mode='train')
        
        # 添加扩散损失
        diffusion_weight = getattr(args, 'diffusion_weight', 0.1)
        total_loss += diffusion_weight * diffusion_loss_dict['total_loss']
        
        meter.update({
            'diff_mse': diffusion_loss_dict['mse_loss'].data,
            'diff_conf': diffusion_loss_dict.get('conf_loss', torch.tensor(0.0)).data
        })
    
    # ========== Common matching triplet loss ==========
    common_rgb_indices = torch.isin(rgb_ids, match_info['common_matched_rgb'])
    common_ir_indices = torch.isin(ir_ids, match_info['common_matched_ir'])
    
    if common_rgb_indices.any() and common_ir_indices.any():
        tri_loss_rgb, tri_loss_ir = _compute_triplet_loss(
            rgb_features, ir_features, rgb_ids, ir_ids,
            common_rgb_indices, common_ir_indices,
            match_info['common_rm'], model, args
        )
        total_loss += tri_loss_rgb + tri_loss_ir
        meter.update({'tri_rgb': tri_loss_rgb.data, 'tri_ir': tri_loss_ir.data})
    
    # ========== Cross-modal alignment (CMA) ==========
    cma.update(bn_features[:2*args.batch_pidnum*args.pid_numsample], 
               bn_features[2*args.batch_pidnum*args.pid_numsample:], rgb_ids, ir_ids)
    
    cma_loss, nan_counter = _compute_cma_loss(
        model, dtd_r2r_cls, dtd_i2i_cls, r2i_cls, i2r_cls, rgb_ids, ir_ids,
        common_rgb_indices, common_ir_indices, match_info['common_rm'], cma, nan_counter
    )
    if cma_loss is not None:
        total_loss += cma_loss
        meter.update({'cma': cma_loss.data})
    
    # ========== Weak supervision for remain pairs ==========
    if epoch >= 30:
        weak_loss, nan_counter = _compute_weak_loss(
            model, bn_features, rgb_ids, ir_ids,
            match_info['remain_matched_rgb'], match_info['remain_matched_ir'],
            match_info['remain_rm'], args, nan_counter
        )
        if weak_loss is not None:
            total_loss += weak_loss
            meter.update({'weak': weak_loss.data})
    
    # ========== Common classifier loss ==========
    r2c_cls = model.classifier3(bn_features)[0][:2*args.batch_pidnum*args.pid_numsample]
    i2c_cls = model.classifier3(bn_features)[0][2*args.batch_pidnum*args.pid_numsample:]
    
    specific_rgb_indices = torch.isin(rgb_ids, match_info['specific_matched_rgb'])
    rgb_indices = specific_rgb_indices ^ common_rgb_indices
    
    if rgb_indices.any():
        rgb_cross_loss = model.pid_criterion(r2c_cls[rgb_indices], match_info['specific_rm'][rgb_ids[rgb_indices]])
        if not torch.isnan(rgb_cross_loss):
            total_loss += rgb_cross_loss
            meter.update({'rgb_cross': rgb_cross_loss.data})
        else:
            nan_counter += 1
    
    ir_cross_loss = model.pid_criterion(i2c_cls, ir_ids)
    total_loss += ir_cross_loss
    meter.update({'ir_cross': ir_cross_loss.data})
    
    return total_loss, nan_counter


def _compute_triplet_loss(rgb_features, ir_features, rgb_ids, ir_ids,
                          common_rgb_indices, common_ir_indices, common_rm, model, args):
    """计算三元组损失"""
    selected_rgb_ids = rgb_ids[common_rgb_indices]
    selected_ir_ids = ir_ids[common_ir_indices]
    
    translated_rgb_label = torch.nonzero(common_rm[selected_rgb_ids])[:, -1]
    translated_ir_label = torch.nonzero(common_rm.T[selected_ir_ids])[:, -1]
    
    selected_rgb_features = rgb_features[common_rgb_indices]
    selected_ir_features = ir_features[common_ir_indices]
    
    matched_rgb_features = torch.cat((selected_rgb_features, ir_features), dim=0)
    matched_ir_features = torch.cat((rgb_features, selected_ir_features), dim=0)
    matched_rgb_labels = torch.cat((translated_rgb_label, ir_ids), dim=0)
    matched_ir_labels = torch.cat((rgb_ids, translated_ir_label), dim=0)
    
    tri_loss_rgb = args.tri_weight * model.tri_criterion(matched_rgb_features, matched_rgb_labels)
    tri_loss_ir = args.tri_weight * model.tri_criterion(matched_ir_features, matched_ir_labels)
    
    return tri_loss_rgb, tri_loss_ir


def _compute_cma_loss(model, dtd_r2r_cls, dtd_i2i_cls, r2i_cls, i2r_cls, rgb_ids, ir_ids,
                      common_rgb_indices, common_ir_indices, common_rm, cma, nan_counter):
    """计算跨模态对齐损失"""
    if not common_rgb_indices.any() or not common_ir_indices.any():
        return None, nan_counter
    
    selected_rgb_ids = rgb_ids[common_rgb_indices]
    selected_ir_ids = ir_ids[common_ir_indices]
    
    translated_rgb_label = torch.nonzero(common_rm[selected_rgb_ids])[:, -1]
    translated_ir_label = torch.nonzero(common_rm.T[selected_ir_ids])[:, -1]
    
    # 信息熵加权
    r2i_entropy = infoEntropy(r2i_cls)
    i2r_entropy = infoEntropy(i2r_cls)
    w_r2i = r2i_entropy / (r2i_entropy + i2r_entropy)
    w_i2r = i2r_entropy / (r2i_entropy + i2r_entropy)
    
    # Memory retrieval
    selected_rgb_memory = cma.vis_memory[translated_ir_label].detach()
    selected_ir_memory = cma.ir_memory[translated_rgb_label].detach()
    
    mem_r2i_cls, _ = model.classifier2(selected_rgb_memory)
    mem_i2r_cls, _ = model.classifier1(selected_ir_memory)
    
    cmo_criterion = torch.nn.MSELoss()
    
    # 计算损失
    r2i_cmo_loss = w_r2i * cmo_criterion(dtd_i2i_cls[common_ir_indices], mem_r2i_cls)
    i2r_cmo_loss = w_i2r * cmo_criterion(dtd_r2r_cls[common_rgb_indices], mem_i2r_cls)
    
    if torch.isnan(r2i_cmo_loss) or torch.isnan(i2r_cmo_loss):
        nan_counter += 1
        return None, nan_counter
    
    return r2i_cmo_loss + i2r_cmo_loss, nan_counter


def _compute_weak_loss(model, bn_features, rgb_ids, ir_ids, remain_matched_rgb, remain_matched_ir,
                       remain_rm, args, nan_counter):
    """计算弱监督损失"""
    remain_rgb_indices = torch.isin(rgb_ids, remain_matched_rgb)
    remain_ir_indices = torch.isin(ir_ids, remain_matched_ir)
    
    if not remain_rgb_indices.any():
        return None, nan_counter
    
    r2c_cls = model.classifier3(bn_features)[0][:2*args.batch_pidnum*args.pid_numsample]
    remain_r2c_cls = r2c_cls[remain_rgb_indices]
    remain_rgb_ids = rgb_ids[remain_rgb_indices]
    
    weak_loss = args.weak_weight * model.weak_criterion(remain_r2c_cls, remain_rm[remain_rgb_ids])
    
    if torch.isnan(weak_loss):
        nan_counter += 1
        return None, nan_counter
    
    return weak_loss, nan_counter


def _baseline_loss(model, meter, r2r_cls, i2i_cls, rgb_features, ir_features, rgb_ids, ir_ids, args):
    """Baseline 模式损失"""
    r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
    i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
    r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
    i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
    
    total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
    meter.update({
        'r2r_id': r2r_id_loss.data,
        'i2i_id': i2i_id_loss.data,
        'r2r_tri': r2r_tri_loss.data,
        'i2i_tri': i2i_tri_loss.data
    })
    return total_loss


def _supervised_loss(model, meter, rgbcls_out, gap_features, args):
    """全监督模式损失"""
    # 需要ground truth标签（此处简化）
    raise NotImplementedError("Supervised learning mode needs ground truth labels")
