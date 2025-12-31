import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityGating(nn.Module):
    def __init__(self, temperature=0.1, momentum=0.9): 
        super().__init__()
        self.base_temperature = temperature
        self.momentum = momentum
        self.scale_factor = nn.Parameter(torch.tensor(0.0))
        
        # Debug statistics
        self.last_stats = {}
    
    def forward(self, features, prototypes, uncertainty=None):
        # Normalize inputs
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        
        # Compute similarity matrix [B, K]
        similarity = torch.mm(features, prototypes.t())
        similarity = (similarity + 1.0) * 5.0
        # Dynamic temperature modulation with uncertainty
        alpha = torch.sigmoid(self.scale_factor)
        
        if uncertainty is not None:
            dynamic_temp = self.base_temperature * (1.0 + alpha * uncertainty.unsqueeze(1))
        else:
            dynamic_temp = self.base_temperature
        
        # Softmax distribution over prototypes
        reliability_dist = F.softmax(similarity / dynamic_temp, dim=1)
        
        # Extract max probability as reliability
        max_reliability, max_indices = reliability_dist.max(dim=1)
        
        # Simple linear scaling: [0, 1] -> [0.05, 1.0]
        final_reliability = 0.05 + 0.95 * max_reliability
        
        # Store statistics for debugging
        with torch.no_grad():
            self.last_stats = {
                'alpha': alpha.item(),
                'effective_temp_mean': dynamic_temp.mean().item() if torch.is_tensor(dynamic_temp) else dynamic_temp,
                'similarity_mean': similarity.mean().item(),
                'similarity_std': similarity.std().item(),
                'reliability_raw_mean': max_reliability.mean().item(),
                'reliability_raw_std': max_reliability.std().item(),
                'reliability_final_mean': final_reliability.mean().item(),
                'reliability_final_std': final_reliability.std().item(),
                'reliability_final_min': final_reliability.min().item(),
                'reliability_final_max': final_reliability.max().item(),
            }
            
            if uncertainty is not None:
                self.last_stats['uncertainty_mean'] = uncertainty.mean().item()
                self.last_stats['uncertainty_std'] = uncertainty.std().item()
        
        return reliability_dist, final_reliability
    
    def get_diagnostics(self):
        """
        Return formatted diagnostics string for logging.
        """
        if not self.last_stats:
            return "🎯 RAGM: No statistics available (not executed yet)"
        
        s = self.last_stats
        lines = []
        lines.append("🎯 RAGM (Reliability-Aware Gating Module):")
        lines.append(f"  ├─ Scale Factor (α):")
        lines.append(f"  │  ├─ Raw: {self.scale_factor.item():.4f}")
        lines.append(f"  │  └─ Sigmoid(α): {s['alpha']:.4f}")
        lines.append(f"  │")
        lines.append(f"  ├─ Temperature:")
        lines.append(f"  │  ├─ Base: {self.base_temperature:.4f}")
        lines.append(f"  │  └─ Effective (Avg): {s['effective_temp_mean']:.4f}")
        lines.append(f"  │")
        lines.append(f"  ├─ Feature-Prototype Similarity:")
        lines.append(f"  │  └─ {s['similarity_mean']:.4f} ± {s['similarity_std']:.4f}")
        lines.append(f"  │")
        lines.append(f"  ├─ Reliability (Raw Max-Prob):")
        lines.append(f"  │  └─ {s['reliability_raw_mean']:.4f} ± {s['reliability_raw_std']:.4f}")
        lines.append(f"  │")
        lines.append(f"  └─ Reliability (Final Scaled):")
        lines.append(f"     ├─ Mean: {s['reliability_final_mean']:.4f} ± {s['reliability_final_std']:.4f}")
        lines.append(f"     └─ Range: [{s['reliability_final_min']:.4f}, {s['reliability_final_max']:.4f}]")
        
        if 'uncertainty_mean' in s:
            lines.append(f"")
            lines.append(f"  [Uncertainty Input]")
            lines.append(f"     └─ {s['uncertainty_mean']:.4f} ± {s['uncertainty_std']:.4f}")
        
        return '\n'.join(lines)
