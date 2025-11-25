import torch
from alive_progress import alive_bar
from utils import MultiItemAverageMeter
from models.loss import OrthogonalLoss, GraphDistillationLoss, ModalityLoss


def train_mgcd(args, model, memory_hub, dataset, epoch, logger):
    """
    PF-MGCD 训练单 Epoch 逻辑
    """
    # 初始化 Loss Functions
    criterion_id = torch.nn.CrossEntropyLoss().to(model.device)
    criterion_orth = OrthogonalLoss().to(model.device)
    criterion_graph = GraphDistillationLoss().to(model.device)
    criterion_mod = ModalityLoss().to(model.device)

    # 使用 Phase 1 优化器 (覆盖了 backbone 和 classifiers)
    optimizer = model.optimizer_phase1

    model.model.train()
    memory_hub.train()

    meter = MultiItemAverageMeter()

    # 获取 DataLoader
    # 原代码逻辑: dataset.get_train_loader() 返回 (rgb_loader, ir_loader)
    rgb_loader, ir_loader = dataset.get_train_loader()
    loader_zip = zip(rgb_loader, ir_loader)

    # 动态权重配置 (参考方案书 Tips: Warm-up)
    lambda_graph = 1.0 if epoch >= 10 else 0.0
    lambda_orth = 0.1
    lambda_mod = 0.1

    with alive_bar(len(rgb_loader), bar='halloween', spinner='dots', force_tty=True,
                   title=f"MGCD Epoch {epoch}") as bar:
        # color_info / ir_info 格式通常是 [index, label, camid, ...]
        for (rgb_imgs, _, rgb_info), (ir_imgs, _, ir_info) in loader_zip:

            # 1. 数据准备
            rgb_imgs = rgb_imgs.to(model.device)
            ir_imgs = ir_imgs.to(model.device)

            # 获取单模态标签 (Ground Truth)
            rgb_pids = rgb_info[:, 1].to(model.device)
            ir_pids = ir_info[:, 1].to(model.device)

            # 构建混合 Batch [RGB, IR]
            imgs = torch.cat([rgb_imgs, ir_imgs], dim=0)
            pids = torch.cat([rgb_pids, ir_pids], dim=0)

            # 构建模态标签: RGB=0, IR=1
            B_rgb = rgb_imgs.size(0)
            B_ir = ir_imgs.size(0)
            mod_labels = torch.cat([
                torch.zeros(B_rgb, dtype=torch.long),
                torch.ones(B_ir, dtype=torch.long)
            ], dim=0).to(model.device)

            optimizer.zero_grad()

            # 2. Forward Pass
            # model.model 指向 PF_MGCD_Model
            # 输出均为 List，长度为 K (num_parts)
            id_feats, mod_feats, id_logits_list, mod_logits_list = model.model(imgs)

            # 3. Memory Graph 交互
            # 将当前 Batch 的身份特征输入记忆库，获取 Soft Targets (图传播结果)
            # soft_targets_list: List of [B, N_classes]
            soft_targets_list = memory_hub(id_feats)

            # 4. 计算损失
            loss_id = 0
            loss_mod = 0
            loss_orth = 0
            loss_graph = 0

            for k in range(len(id_feats)):
                # A. 身份分类损失 (Cross Entropy)
                loss_id += criterion_id(id_logits_list[k], pids)

                # B. 模态判别损失
                loss_mod += criterion_mod([mod_logits_list[k]], mod_labels)

                # C. 正交解耦损失
                loss_orth += criterion_orth([id_feats[k]], [mod_feats[k]])

                # D. 图蒸馏损失 (Warm-up后开启)
                if lambda_graph > 0:
                    loss_graph += criterion_graph([id_logits_list[k]], [soft_targets_list[k]])

            # 总损失
            loss_total = loss_id + lambda_mod * loss_mod + lambda_orth * loss_orth + lambda_graph * loss_graph

            loss_total.backward()
            optimizer.step()

            # 5. 更新记忆库 (Momentum Update)
            # 使用当前 Batch 的特征更新对应 ID 的原型
            memory_hub.update(id_feats, pids)

            # 6. 记录日志
            meter.update({
                'L_tot': loss_total.data,
                'L_id': loss_id.data,
                'L_gr': loss_graph.data if lambda_graph > 0 else 0,
                'L_or': loss_orth.data,
                'L_mo': loss_mod.data
            })

            bar()

    return meter.get_val(), meter.get_str()