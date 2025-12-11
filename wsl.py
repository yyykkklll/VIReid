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
    Cross modal Match Aggregation
    Modified to support CLIP Referee and Sinkhorn Matching
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.not_saved = True
        self.num_classes = args.num_classes
        self.T = args.temperature # softmax temperature
        self.sigma = args.sigma # momentum update factor
        self.args = args 
        
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))
        
        # CLIP 特征临时存储
        self.vis_clip_feats = None
        self.ir_clip_feats = None

    @torch.no_grad()
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None, clip_rgb=None, clip_ir=None):
    # vis: vis sample(v2i scores or vis features) ir: ir sample
        self.mode = mode
        self.not_saved = False
        if self.mode != 'scores' and self.mode != 'features':
            raise ValueError('invalid mode!')
        elif self.mode == 'scores': # predict scores
            vis = torch.nn.functional.softmax(self.T*vis,dim=1)
            ir = torch.nn.functional.softmax(self.T*ir,dim=1)
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
                # .any() check True in bool tensor
                if rgb_mask.any():
                    rgb_selected = rgb_features[rgb_mask]
                    self.vis_memory[label] = rgb_selected.mean(dim=0)
                
                if ir_mask.any():
                    ir_selected = ir_features[ir_mask]
                    self.ir_memory[label] = ir_selected.mean(dim=0)
        
        # [NEW] Save CLIP features for matching
        if clip_rgb is not None and clip_ir is not None:
            self.vis_clip_feats = clip_rgb.detach().cpu()
            self.ir_clip_feats = clip_ir.detach().cpu()
        ################################
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

    def get_label(self, epoch=None):
        if self.not_saved:# pass if 
            pass
        else:
            print('get match labels')
            
            # [NEW] Sinkhorn Matching Branch
            if hasattr(self.args, 'use_sinkhorn') and self.args.use_sinkhorn:
                print("Using Sinkhorn Global Matching...")
                return self._get_label_sinkhorn()

            # Original Greedy Matching
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict = self._get_label(dists,'dist')

            elif self.mode == 'scores':
                v2i_dict, _ = self._get_label(self.vis,'rgb')
                i2v_dict, _ = self._get_label(self.ir,'ir')
                self.v2i = v2i_dict
                self.i2v = i2v_dict
            return v2i_dict, i2v_dict

    # [NEW] Sinkhorn Algorithm Implementation
    def _get_label_sinkhorn(self):
        # 1. 计算专家相似度 (基于分类分数)
        score_rgb = torch.from_numpy(self.vis).to(self.device)
        score_ir = torch.from_numpy(self.ir).to(self.device)
        
        # Normalize
        score_rgb = torch.nn.functional.normalize(score_rgb, dim=1)
        score_ir = torch.nn.functional.normalize(score_ir, dim=1)
        
        sim_expert = torch.matmul(score_rgb, score_ir.T) # [N_vis, N_ir]

        # 2. 计算 CLIP 语义相似度 (如果启用了 CLIP)
        if self.vis_clip_feats is not None and self.ir_clip_feats is not None:
            clip_rgb = self.vis_clip_feats.to(self.device)
            clip_ir = self.ir_clip_feats.to(self.device)
            
            clip_rgb = torch.nn.functional.normalize(clip_rgb, dim=1)
            clip_ir = torch.nn.functional.normalize(clip_ir, dim=1)
            
            sim_clip = torch.matmul(clip_rgb, clip_ir.T)
            
            # 融合权重
            w_clip = getattr(self.args, 'w_clip', 0.3)
            sim_final = (1 - w_clip) * sim_expert + w_clip * sim_clip
        else:
            sim_final = sim_expert

        # 3. 构建代价矩阵 (Cost = 1 - Similarity)
        # Q = exp(Sim / epsilon)
        epsilon = getattr(self.args, 'sinkhorn_reg', 0.05)
        
        # Sinkhorn Iteration
        Q = torch.exp(sim_final / epsilon)
        
        for _ in range(50): 
            Q /= Q.sum(dim=0, keepdim=True) + 1e-8
            Q /= Q.sum(dim=1, keepdim=True) + 1e-8
            
        # 4. 从 Q 生成匹配字典
        Q = Q.cpu().numpy()
        
        # 生成 v2i
        v2i = OrderedDict()
        max_j = np.argmax(Q, axis=1)
        for i, j in enumerate(max_j):
            # 双向验证
            if np.argmax(Q[:, j]) == i:
                real_id_rgb = self.rgb_ids[i].item()
                real_id_ir = self.ir_ids[j].item()
                if real_id_rgb not in v2i:
                     v2i[real_id_rgb] = real_id_ir

        # 生成 i2v
        i2v = OrderedDict()
        max_i = np.argmax(Q, axis=0)
        for j, i in enumerate(max_i):
             if np.argmax(Q[i, :]) == j:
                real_id_rgb = self.rgb_ids[i].item()
                real_id_ir = self.ir_ids[j].item()
                if real_id_ir not in i2v:
                    i2v[real_id_ir] = real_id_rgb
                    
        return v2i, i2v

    def _get_label(self,dists,mode):
        sample_rate = 1
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]# flat to 1d and sort
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)# sort index return to 2d, like ([0,1,2],[1,2,0])
        idx1, idx2 = sorted_2d[0], sorted_2d[1]# sorted idx of dim0 and dim1
        dists = dists[idx1, idx2]
        idx_length = int(np.ceil(sample_rate*dists.shape[0]/self.num_classes))
        dists = dists[:idx_length]

        if mode=='dist': # multiply the instance features of the two modalities
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                            np.array(self.ir_ids)[idx2[:idx_length]])]
            
        elif mode=='rgb': # classify score of RGB (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]

        elif mode=='ir': # classify score of IR (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.ir_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]
        else:
            raise AttributeError('invalid mode!')
        convert_label_cnt = Counter(convert_label)
        convert_label_cnt_sorted = sorted(convert_label_cnt.items(),key = lambda x:x[1],reverse = True)
        length = len(convert_label_cnt_sorted)
        lambda_cm=0.1
        in_rgb_label=[]
        in_ir_label=[]
        v2i = OrderedDict()
        i2v = OrderedDict()

        length_ratio = 1
        for i in range(int(length*length_ratio)):
            key = convert_label_cnt_sorted[i][0] 
            value = convert_label_cnt_sorted[i][1]
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
            v2i[key[0]] = key[1]
            i2v[key[1]] = key[0]
            
        return v2i, i2v 

    # [MODIFIED] 增加 clip_model 参数
    def extract(self, args, model, dataset, clip_model=None):
        '''
        Output: BN_features, labels, cls
        '''
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader() 
        with torch.no_grad():
            rgb_features, rgb_labels, rgb_gt, r2i_cls, rgb_idx, rgb_clip = self._extract_feature(model, rgb_loader,'rgb', clip_model)
            ir_features, ir_labels, ir_gt, i2r_cls, ir_idx, ir_clip = self._extract_feature(model, ir_loader,'ir', clip_model)

        # Pass clip features to save
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx,\
                 ir_idx, 'scores', rgb_features, ir_features, clip_rgb=rgb_clip, clip_ir=ir_clip)
        
    def _extract_feature(self, model, loader, modal, clip_model=None):
        print('extracting {} features'.format(modal))

        saved_features, saved_labels, saved_cls= None, None, None
        saved_gts, saved_idx= None, None
        saved_clip_feats = None

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
            
            # [FIXED] Extract CLIP features correctly
            batch_clip_feats = None
            if clip_model is not None:
                # 1. 编码图像，得到特征图 x4 (N, C, H, W) 或元组
                out = clip_model.encode_image(imgs)
                
                # 2. 获取全局特征
                # 该仓库的 ResNet 实现比较特殊，Visual Encoder 返回的是 Feature Map (x4)
                # 而我们需要的是 attnpool 后的投影特征 (xproj)
                # 幸好 ModifiedResNet 保留了 attnpool 层
                
                if hasattr(clip_model.visual, 'attnpool'):
                    # 如果是 ResNet 结构
                    feat_map = out 
                    # [CRITICAL FIX] attnpool returns (Seq, Batch, Dim), we need index [0]
                    clip_emb = clip_model.visual.attnpool(feat_map)[0] 
                else:
                    # 如果是 ViT 结构 (encode_image 返回 (x11, x12, xproj))
                    if isinstance(out, tuple):
                        # xproj in ViT usually is (Seq, Batch, Dim) too?
                        # Usually ViT output in this codebase is xproj = x12 @ proj.
                        # Let's assume safe fallback to slice 0 just in case it is seq.
                        # But for RN50 path, the above if branch handles it.
                        clip_emb = out[-1]
                        if len(clip_emb.shape) == 3:
                             clip_emb = clip_emb[0]
                    else:
                        clip_emb = out

                batch_clip_feats = clip_emb.detach().cpu()

            if saved_features is None: 
                saved_features, saved_labels, saved_cls, saved_idx = bn_features, labels, cls, idx
                saved_gts = gts
                if batch_clip_feats is not None:
                    saved_clip_feats = batch_clip_feats
            else:
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)

                saved_gts = torch.cat((saved_gts, gts), dim=0)
                if batch_clip_feats is not None:
                    # [FIXED] Now shapes match along dim 0 (Batch dimension)
                    saved_clip_feats = torch.cat((saved_clip_feats, batch_clip_feats), dim=0)
                    
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx, saved_clip_feats