import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeGuidance(nn.Module):
    def __init__(self, num_classes=206, feat_dim=2048, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 120
        
        # Debug statistics
        self.last_guidance_loss = 0.0
        self.last_guidance_weight = 0.0
    
    def set_epoch(self, epoch, total_epochs=120):
        """Update current epoch for dynamic guidance scheduling"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def get_guidance_weight(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
        
        if epoch < 20:
            weight = 0.5
        elif epoch < 80:
            weight = 0.4
        else:
            # Linear decay from 0.4 to 0.2
            decay_ratio = (epoch - 80) / (self.total_epochs - 80 + 1e-6)
            weight = 0.4 - 0.2 * max(0.0, min(1.0, decay_ratio))
        
        self.last_guidance_weight = weight
        return weight
    
    def compute_guidance_loss(self, feat_generated, prototypes, labels, temperature=0.07):
        # Cosine similarity as logits
        # feat: [B, D], prototypes: [C, D] -> logits: [B, C]
        logits = torch.matmul(F.normalize(feat_generated, dim=1),
                             F.normalize(prototypes, dim=1).t())
        logits = logits / temperature
        
        loss = F.cross_entropy(logits, labels)
        self.last_guidance_loss = loss.item()
        
        return loss
    
    def get_diagnostics(self):
        """
        Return formatted diagnostics string for logging.
        """
        lines = []
        lines.append("🧭 Prototype Guidance:")
        lines.append(f"  ├─ Current Epoch: {self.current_epoch}/{self.total_epochs}")
        lines.append(f"  ├─ Guidance Weight: {self.last_guidance_weight:.4f}")
        lines.append(f"  └─ Last Guidance Loss: {self.last_guidance_loss:.4f}")
        
        # Explain the schedule
        if self.current_epoch < 20:
            lines.append(f"  [Schedule: Early Stage - High Guidance (0.5)]")
        elif self.current_epoch < 80:
            lines.append(f"  [Schedule: Mid Stage - Moderate Guidance (0.4)]")
        else:
            lines.append(f"  [Schedule: Late Stage - Decaying Guidance (0.4→0.2)]")
        
        return '\n'.join(lines)
