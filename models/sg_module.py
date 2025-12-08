import torch
import torch.nn as nn
import numpy as np

class SGM(nn.Module):
    def __init__(self, args):
        super(SGM, self).__init__()
        self.num_classes = args.num_classes
        self.T = args.temperature 
        self.sigma = args.sigma 
        
        feat_dim = getattr(args, 'feat_dim', 768) 
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, feat_dim))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, feat_dim))
        
        self.has_extracted = False

    def extract(self, args, model, dataset):
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader()
        
        target_device = next(model.model.parameters()).device
        if self.vis_memory.device != target_device:
            self.to(target_device)
        
        print('>>> [SGM] Extracting features...')
        with torch.no_grad():
            rgb_features, rgb_labels, _ = self._extract_feature(model, rgb_loader)
            ir_features, ir_labels, _ = self._extract_feature(model, ir_loader)

        self._update_memory(rgb_features, rgb_labels, 'rgb')
        self._update_memory(ir_features, ir_labels, 'ir')
        self.has_extracted = True
        
    def _extract_feature(self, model, loader):
        features_list = []
        labels_list = []
        idx_list = []
        device = next(model.model.parameters()).device

        for imgs, infos in loader:
            imgs = imgs.to(device)
            pids = infos[:, 1].to(device)
            idxs = infos[:, 0].to(device)
            _, bn_feat = model.model(imgs)
            bn_feat = torch.nn.functional.normalize(bn_feat, p=2, dim=1)
            features_list.append(bn_feat)
            labels_list.append(pids)
            idx_list.append(idxs)
            
        return torch.cat(features_list, 0), torch.cat(labels_list, 0), torch.cat(idx_list, 0)

    def _update_memory(self, features, labels, modal):
        unique_labels = torch.unique(labels)
        target_memory = self.vis_memory if modal == 'rgb' else self.ir_memory
        for label in unique_labels:
            mask = (labels == label)
            class_feats = features[mask]
            mean_feat = class_feats.mean(dim=0)
            target_memory[label] = (1 - self.sigma) * target_memory[label] + self.sigma * mean_feat

    def sinkhorn(self, cost_matrix, epsilon=0.1, iterations=5):
        # 使用 Log-domain 计算，防止数值不稳定
        # cost_matrix: (N, M)
        C = cost_matrix
        num_r, num_c = C.shape
        
        mu = torch.zeros(num_r, device=C.device)
        nu = torch.zeros(num_c, device=C.device)
        
        for _ in range(iterations):
            # 行归一化 (Log domain)
            M = -C / epsilon + nu.unsqueeze(0)
            mu = -torch.logsumexp(M, dim=1)
            # 列归一化 (Log domain)
            M = -C / epsilon + mu.unsqueeze(1)
            nu = -torch.logsumexp(M, dim=0)

        U = mu.unsqueeze(1) + nu.unsqueeze(0)
        P = torch.exp(U - C/epsilon)
        return P

    def mining_structure_relations(self, epoch=None):
        if not self.has_extracted:
            return {}, {}
            
        print(f'>>> [SGM] Mining Structure Relations (Sinkhorn + Bi-Check) Epoch {epoch}')
        
        # 1. 计算欧氏距离平方 (比 Cosine 更适合 OT)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # 因为特征已经归一化，||a||=1, ||b||=1, 所以 Dist = 2 - 2 * Cosine
        sim_matrix = torch.mm(self.vis_memory, self.ir_memory.t())
        dist_matrix = 2 - 2 * sim_matrix
        
        # 2. Sinkhorn 全局优化
        with torch.no_grad():
            P = self.sinkhorn(dist_matrix, epsilon=0.1, iterations=5)
        
        P_cpu = P.cpu().numpy()
        
        # 3. 双向匹配 + 严格阈值
        v2i = {}
        i2v = {}
        
        # 阈值策略：至少要比均匀分布 (1/N) 高出一定倍数
        # RegDB N=206, 1/N ≈ 0.005. Threshold = 0.01
        threshold = (1.0 / self.num_classes) * 2.0 
        
        # 3.1 提取潜在匹配
        matches_r_idx = np.argmax(P_cpu, axis=1) # RGB -> IR
        matches_r_val = np.max(P_cpu, axis=1)
        
        matches_i_idx = np.argmax(P_cpu, axis=0) # IR -> RGB
        matches_i_val = np.max(P_cpu, axis=0)
        
        # 3.2 填充字典 (只保留高置信度)
        for r_id in range(self.num_classes):
            target_ir = matches_r_idx[r_id]
            # 双向验证: RGB->IR 是 target, 且 IR->RGB 必须回指 RGB
            if matches_r_val[r_id] > threshold and matches_i_idx[target_ir] == r_id:
                v2i[r_id] = target_ir
        
        for i_id in range(self.num_classes):
            target_rgb = matches_i_idx[i_id]
            # 双向验证
            if matches_i_val[i_id] > threshold and matches_r_idx[target_rgb] == i_id:
                i2v[i_id] = target_rgb
            
        print(f'>>> [SGM] Found {len(v2i)} confident RGB->IR pairs out of {self.num_classes}')
        return v2i, i2v