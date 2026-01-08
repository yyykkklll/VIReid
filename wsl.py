import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from sklearn.preprocessing import normalize
import time
import pickle
from utils import fliplr
class CMA(nn.Module):
    '''
    Cross modal Match Aggregation with Sinkhorn
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.not_saved = True
        self.num_classes = args.num_classes
        self.T = args.temperature 
        self.sigma = args.sigma 
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))
        
        # Sinkhorn params
        self.epsilon = 0.05 # Smoothing parameter
        self.sinkhorn_iters = 10
        self.confidence_threshold = 0.0 # Return all or filter? Plan says 0.5 usually

    @torch.no_grad()
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None):
        self.mode = mode
        self.not_saved = False
        
        ###############################
        # save features in memory bank
        if rgb_features is not None and ir_features is not None:
            # Prepare empty memory banks on the device
            self.vis_memory = self.vis_memory.to(self.device)
            self.ir_memory = self.ir_memory.to(self.device)
            
            # Get unique labels and process RGB and IR features
            label_set = torch.unique(rgb_ids)
            
            for label in label_set:
                # Select RGB features for the current label
                rgb_mask = (rgb_ids == label)
                ir_mask = (ir_ids == label)
                
                if rgb_mask.any():
                    rgb_selected = rgb_features[rgb_mask]
                    self.vis_memory[label] = rgb_selected.mean(dim=0)
                
                if ir_mask.any():
                    ir_selected = ir_features[ir_mask]
                    self.ir_memory[label] = ir_selected.mean(dim=0)
        ################################
        
        # We store these just in case, but we mainly use memory bank for Sinkhorn
        vis = vis.detach().cpu().numpy()
        ir = ir.detach().cpu().numpy()
        rgb_ids, ir_ids = rgb_ids.cpu(), ir_ids.cpu()
            
        self.vis, self.ir = vis, ir
        self.rgb_ids, self.ir_ids = rgb_ids, ir_ids
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx
        
    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        rgb_set = torch.unique(rgb_labels)
        ir_set = torch.unique(ir_labels)
        for i in rgb_set:
            rgb_mask = (rgb_labels == i)
            selected_rgb = rgb_feats[rgb_mask].mean(dim=0)
            self.vis_memory[i] = (1-self.sigma)*self.vis_memory[i] + self.sigma * selected_rgb
        for i in ir_set:
            ir_mask = (ir_labels == i)
            selected_ir = ir_feats[ir_mask].mean(dim=0)
            self.ir_memory[i] = (1-self.sigma)*self.ir_memory[i] + self.sigma * selected_ir

    def get_label(self, epoch=None, debug_logger=None):
        if self.not_saved:
            if debug_logger:
                debug_logger("CMA.get_label: Features not saved yet. Returning empty.")
            return []
        else:
            print('Calculating Global Optimal Assignment (Sinkhorn)...')
            if debug_logger:
                debug_logger(f"CMA.get_label: Starting Sinkhorn Matching for Epoch {epoch}")
            return self._sinkhorn_matching(debug_logger)

    def _sinkhorn_matching(self, debug_logger=None):
        # 1. Compute Cost Matrix
        # Using 1 - Cosine Similarity
        # vis_memory: (N, D), ir_memory: (N, D)
        # Normalize first
        vis_norm = torch.nn.functional.normalize(self.vis_memory, p=2, dim=1)
        ir_norm = torch.nn.functional.normalize(self.ir_memory, p=2, dim=1)
        
        # Cosine Similarity: (N, N)
        sim_matrix = torch.matmul(vis_norm, ir_norm.t())
        
        # Cost = 1 - Sim (or Euclidean, but 1-Sim is standard for cosine)
        # Using Euclidean might be better for Sinkhorn if features are not normalized, 
        # but here they are.
        cost_matrix = 1.0 - sim_matrix
        
        if debug_logger:
            debug_logger(f"Sinkhorn Cost Matrix Stats: Min={cost_matrix.min().item():.4f}, Max={cost_matrix.max().item():.4f}, Mean={cost_matrix.mean().item():.4f}")
        
        # 2. Sinkhorn Algorithm
        # P = exp(-C / epsilon)
        P = torch.exp(-cost_matrix / self.epsilon)
        
        # Iterative Normalization
        for i in range(self.sinkhorn_iters):
            # Row normalization (RGB needs to match someone)
            P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
            # Column normalization (IR needs to receive someone)
            P = P / (P.sum(dim=0, keepdim=True) + 1e-8)
            
        # P is now a doubly stochastic matrix (or close to it)
        # P_ij is probability that RGB_i matches IR_j
        
        if debug_logger:
            debug_logger(f"Sinkhorn P Matrix Stats (Final): Min={P.min().item():.6f}, Max={P.max().item():.6f}, Mean={P.mean().item():.6f}")

        # 3. Generate Match List (Optimized: Cycle Consistency + Non-linear Weighting)
        match_list = []
        rows, cols = P.shape
        
        # RGB -> IR Best Matches
        # max_probs_r2i: (N_rgb,), indices_r2i: (N_rgb,)
        max_probs_r2i, indices_r2i = torch.max(P, dim=1)
        
        # IR -> RGB Best Matches (for Cycle Consistency)
        # max_probs_i2r: (N_ir,), indices_i2r: (N_ir,)
        max_probs_i2r, indices_i2r = torch.max(P, dim=0)
        
        min_threshold = 1e-4
        count_reliable = 0
        count_weak = 0
        
        for r in range(rows):
            c = indices_r2i[r].item()
            score = max_probs_r2i[r].item()
            
            if score > min_threshold:
                # Check Cycle Consistency: Does IR column 'c' also pick RGB row 'r' as best?
                is_cycle_consistent = (indices_i2r[c].item() == r)
                
                final_weight = score
                if is_cycle_consistent:
                    # Reliable match: Keep linear weight (or boost slightly?)
                    # score is enough.
                    count_reliable += 1
                else:
                    # Weak/One-way match: Penalize heavily
                    # Square the score to suppress low-confidence noise
                    # e.g., 0.3 -> 0.09, 0.8 -> 0.64
                    final_weight = score ** 2
                    count_weak += 1
                
                match_list.append((r, c, final_weight))

        if debug_logger:
            debug_logger(f"Sinkhorn Strategy: Cycle Consistency + Square Penalty. Total: {len(match_list)}")
            debug_logger(f"Reliable (Cycle): {count_reliable}, Weak (One-way): {count_weak}")
            if len(match_list) > 0:
                weights = [m[2] for m in match_list]
                debug_logger(f"Weight Stats: Min={min(weights):.4f}, Max={max(weights):.4f}, Mean={sum(weights)/len(weights):.4f}")

        print(f"Sinkhorn matches: {len(match_list)} (Reliable: {count_reliable}, Weak: {count_weak})")
        return match_list


    def extract(self, args, model, dataset):
        '''
        Output: BN_features, labels, cls
        '''
        # save epoch
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader() 
        with torch.no_grad():
            
            rgb_features, rgb_labels, rgb_gt, r2i_cls, rgb_idx = self._extract_feature(model, rgb_loader,'rgb')
            ir_features, ir_labels, ir_gt, i2r_cls, ir_idx = self._extract_feature(model, ir_loader,'ir')

        # # //match by cls and save features to memory bank
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx,
                 ir_idx, 'scores', rgb_features, ir_features)
        
    def _extract_feature(self, model, loader, modal):

        print('extracting {} features'.format(modal))

        saved_features, saved_labels, saved_cls= None, None, None
        saved_gts, saved_idx= None, None
        for imgs_list, infos in loader:
            labels = infos[:,1]
            idx = infos[:,0]
            gts = infos[:,-1].to(model.device)
            if imgs_list.__class__.__name__ != 'list':
                imgs = imgs_list
                imgs, labels, idx = \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            else:
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)

                imgs = torch.cat((ori_imgs,ca_imgs),dim=0)
                labels = torch.cat((labels,labels),dim=0)
                idx = torch.cat((idx,idx),dim=0)
                gts= torch.cat((gts,gts),dim=0).to(model.device)
                imgs, labels, idx= \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            _, bn_features = model.model(imgs) # _:gap feature

            if modal == 'rgb':
                cls, l2_features = model.classifier2(bn_features)
            elif modal == 'ir':
                cls, l2_features = model.classifier1(bn_features)
            l2_features = l2_features.detach().cpu()

            if saved_features is None: 
                # saved_features, saved_labels, saved_cls, saved_idx = l2_features, labels, cls, idx
                saved_features, saved_labels, saved_cls, saved_idx = bn_features, labels, cls, idx

                saved_gts = gts
            else:
                # saved_features = torch.cat((saved_features, l2_features), dim=0)
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)

                saved_gts = torch.cat((saved_gts, gts), dim=0)
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx
