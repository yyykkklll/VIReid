import torch.nn as nn
import torch.nn.functional as F

from .isg_dm import ISG_DM
from .pcb_resnet import PCB_ResNet


class PF_MGCD_Model(nn.Module):
    def __init__(self, args):
        super(PF_MGCD_Model, self).__init__()
        self.num_parts = 6  # 默认 K=6
        self.num_classes = args.num_classes

        # 1. Backbone (No GAP)
        self.backbone = PCB_ResNet(pretrained=True)

        # 2. ISG-DM Modules (Shared or Independent? 方案书 imply independent processing per part but implementation can share weights)
        # 为了参数效率，这里我们可以让 ISG-DM 共享权重，或者独立。
        # 通常 PCB 中 classifier 是独立的，feature embedding 可以共享。
        # 这里我们定义一个 ISG-DM，对所有 part 复用。
        self.disentangler = ISG_DM(in_dim=2048, out_dim=256)

        # 3. Classifiers (Part-specific Identity Classifiers)
        self.id_classifiers = nn.ModuleList([
            nn.Linear(256, self.num_classes, bias=False)
            for _ in range(self.num_parts)
        ])

        # 4. Modality Classifiers (Part-specific or Shared?)
        # 方案书 746 行: self.mod_classifier. 似乎是共享的。
        self.mod_classifier = nn.Linear(256, 2, bias=False)

        # 参数初始化
        for m in self.id_classifiers:
            nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.mod_classifier.weight, std=0.001)

    def forward(self, x):
        # 1. Backbone & Slicing
        # x: [B, 3, H, W] -> feat_map: [B, 2048, 24, 8] (Example size)
        feat_map = self.backbone(x)
        B, C, H, W = feat_map.shape

        # 2. Part Pooling (PCB Slicing)
        # 假设 H 是 num_parts 的倍数。 ResNet50 no-stride output H is usually 24 (384/16).
        # 24 / 6 = 4.
        part_h = H // self.num_parts

        id_feats = []
        mod_feats = []
        id_logits_list = []
        mod_logits_list = []  # 方案书只提了一个mod loss，这里我们每个part都算

        for i in range(self.num_parts):
            # Slicing: [B, C, h_part, W]
            part_tensor = feat_map[:, :, i * part_h: (i + 1) * part_h, :]
            # Pooling: [B, C]
            part_vec = F.avg_pool2d(part_tensor, (part_tensor.size(2), part_tensor.size(3))).squeeze()

            if len(part_vec.shape) == 1:  # Handle batch size 1
                part_vec = part_vec.unsqueeze(0)

            # 3. Disentanglement
            f_id, f_mod = self.disentangler(part_vec)

            # 4. Prediction
            id_logits = self.id_classifiers[i](f_id)
            mod_logits = self.mod_classifier(f_mod)

            id_feats.append(f_id)
            mod_feats.append(f_mod)
            id_logits_list.append(id_logits)
            mod_logits_list.append(mod_logits)

        return id_feats, mod_feats, id_logits_list, mod_logits_list