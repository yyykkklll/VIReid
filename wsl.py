"""
Cross-Modal Match Aggregation (CMA)
跨模态匹配聚合模块 - 支持扩散桥生成的特征
"""
import torch
import torch.nn as nn
import numpy as np
from collections import Counter, OrderedDict


class CMA(nn.Module):
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.not_saved = True
        self.num_classes = args.num_classes
        self.T = args.temperature  # Softmax temperature
        self.sigma = args.sigma  # Momentum update factor
        
        # Memory banks for visible and infrared modalities
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, 2048))

    @torch.no_grad()
    def save(self, vis, ir, rgb_ids, ir_ids, rgb_idx, ir_idx, mode, rgb_features=None, ir_features=None):
        """
        保存特征和分类分数
        Args:
            vis: RGB 模态数据（分数或特征）
            ir: IR 模态数据（分数或特征）
            mode: 'scores' 或 'features'
            rgb_features: RGB 特征（用于初始化 memory bank）
            ir_features: IR 特征（用于初始化 memory bank）
        """
        self.mode = mode
        self.not_saved = False
        
        if mode not in ['scores', 'features']:
            raise ValueError(f'Invalid mode: {mode}')
        
        # 如果是分数模式，应用 softmax
        if mode == 'scores':
            vis = torch.nn.functional.softmax(self.T * vis, dim=1)
            ir = torch.nn.functional.softmax(self.T * ir, dim=1)
        
        # ========== 初始化 Memory Bank ==========
        if rgb_features is not None and ir_features is not None:
            self._init_memory_bank(rgb_features, ir_features, rgb_ids, ir_ids)
        # ========================================
        
        # 转换为 numpy 用于匹配
        self.vis = vis.detach().cpu().numpy()
        self.ir = ir.detach().cpu().numpy()
        self.rgb_ids = rgb_ids.cpu()
        self.ir_ids = ir_ids.cpu()
        self.rgb_idx = rgb_idx
        self.ir_idx = ir_idx
    
    def _init_memory_bank(self, rgb_features, ir_features, rgb_ids, ir_ids):
        """初始化 Memory Bank（使用平均特征）"""
        self.vis_memory = self.vis_memory.to(self.device)
        self.ir_memory = self.ir_memory.to(self.device)
        
        # 对每个类别计算平均特征
        unique_rgb_ids = torch.unique(rgb_ids)
        unique_ir_ids = torch.unique(ir_ids)
        
        for label in unique_rgb_ids:
            mask = (rgb_ids == label)
            if mask.any():
                self.vis_memory[label] = rgb_features[mask].mean(dim=0)
        
        for label in unique_ir_ids:
            mask = (ir_ids == label)
            if mask.any():
                self.ir_memory[label] = ir_features[mask].mean(dim=0)

    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        """
        动量更新 Memory Bank
        Args:
            rgb_feats: RGB 特征
            ir_feats: IR 特征
            rgb_labels: RGB 标签
            ir_labels: IR 标签
        """
        for label in torch.unique(rgb_labels):
            mask = (rgb_labels == label)
            selected_rgb = rgb_feats[mask].mean(dim=0)
            self.vis_memory[label] = (1 - self.sigma) * self.vis_memory[label] + self.sigma * selected_rgb
        
        for label in torch.unique(ir_labels):
            mask = (ir_labels == label)
            selected_ir = ir_feats[mask].mean(dim=0)
            self.ir_memory[label] = (1 - self.sigma) * self.ir_memory[label] + self.sigma * selected_ir

    def get_label(self, epoch=None):
        """
        获取跨模态匹配标签
        Returns:
            v2i_dict: RGB → IR 匹配字典
            i2v_dict: IR → RGB 匹配字典
        """
        if self.not_saved:
            return {}, {}
        
        print('Getting match labels...')
        
        if self.mode == 'features':
            dists = np.matmul(self.vis, self.ir.T)
            v2i_dict, i2v_dict = self._get_label(dists, 'dist')
        elif self.mode == 'scores':
            v2i_dict, _ = self._get_label(self.vis, 'rgb')
            i2v_dict, _ = self._get_label(self.ir, 'ir')
        
        return v2i_dict, i2v_dict
    
    def _get_label(self, dists, mode):
        """
        从距离/分数矩阵生成匹配标签
        Args:
            dists: 距离或分数矩阵
            mode: 'dist', 'rgb', 或 'ir'
        Returns:
            v2i: RGB → IR 匹配
            i2v: IR → RGB 匹配
        """
        sample_rate = 1
        dists_shape = dists.shape
        
        # 展平并排序
        sorted_1d = np.argsort(dists, axis=None)[::-1]
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)
        idx1, idx2 = sorted_2d[0], sorted_2d[1]
        
        # 采样
        idx_length = int(np.ceil(sample_rate * len(sorted_1d) / self.num_classes))
        dists = dists[idx1[:idx_length], idx2[:idx_length]]
        
        # 构建候选匹配对
        if mode == 'dist':
            convert_label = [(int(self.rgb_ids[i]), int(self.ir_ids[j])) 
                            for i, j in zip(idx1[:idx_length], idx2[:idx_length])]
        elif mode == 'rgb':
            convert_label = [(int(self.rgb_ids[i]), int(j)) 
                            for i, j in zip(idx1[:idx_length], idx2[:idx_length])]
        elif mode == 'ir':
            convert_label = [(int(self.ir_ids[i]), int(j)) 
                            for i, j in zip(idx1[:idx_length], idx2[:idx_length])]
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
        # 统计并排序
        convert_label_cnt = Counter(convert_label)
        convert_label_sorted = sorted(convert_label_cnt.items(), key=lambda x: x[1], reverse=True)
        
        # 贪心匹配（避免一对多）
        v2i, i2v = OrderedDict(), OrderedDict()
        matched_rgb, matched_ir = set(), set()
        
        for (rgb_id, ir_id), count in convert_label_sorted:
            if rgb_id in matched_rgb or ir_id in matched_ir:
                continue
            v2i[rgb_id] = ir_id
            i2v[ir_id] = rgb_id
            matched_rgb.add(rgb_id)
            matched_ir.add(ir_id)
        
        return v2i, i2v

    def extract(self, args, model, dataset):
        """
        提取特征并保存用于匹配
        """
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader()
        
        with torch.no_grad():
            rgb_features, rgb_labels, _, r2i_cls, rgb_idx = self._extract_feature(model, rgb_loader, 'rgb')
            ir_features, ir_labels, _, i2r_cls, ir_idx = self._extract_feature(model, ir_loader, 'ir')
        
        # 保存用于匹配
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx, ir_idx, 'scores', 
                  rgb_features, ir_features)
    
    def _extract_feature(self, model, loader, modal):
        """
        提取单个模态的特征
        Args:
            model: 模型
            loader: 数据加载器
            modal: 'rgb' 或 'ir'
        Returns:
            features, labels, gts, cls, idx
        """
        print(f'Extracting {modal} features...')
        
        saved_features, saved_labels, saved_gts, saved_cls, saved_idx = None, None, None, None, None
        
        for imgs_list, infos in loader:
            labels = infos[:, 1]
            idx = infos[:, 0]
            gts = infos[:, -1].to(model.device)
            
            # 处理输入图像
            if not isinstance(imgs_list, list):
                imgs = imgs_list.to(model.device)
            else:
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)
                imgs = torch.cat((ori_imgs, ca_imgs), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                idx = torch.cat((idx, idx), dim=0)
                gts = torch.cat((gts, gts), dim=0)
            
            imgs = imgs.to(model.device)
            labels = labels.to(model.device)
            idx = idx.to(model.device)
            
            # 特征提取
            _, bn_features = model.model(imgs)
            
            # 跨模态分类
            if modal == 'rgb':
                cls, _ = model.classifier2(bn_features)
            elif modal == 'ir':
                cls, _ = model.classifier1(bn_features)
            
            # 累积
            if saved_features is None:
                saved_features, saved_labels, saved_cls, saved_idx, saved_gts = \
                    bn_features, labels, cls, idx, gts
            else:
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)
                saved_gts = torch.cat((saved_gts, gts), dim=0)
        
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx
