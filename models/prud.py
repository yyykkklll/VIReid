import torch
import torch.nn as nn
import torch.nn.functional as F

class PRUD(nn.Module):
    def __init__(self, num_classes=206, rectification_threshold=0.3, 
                 temp=0.1, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.rectification_threshold = rectification_threshold # Similarity threshold for valid prototypes
        self.temp = temp # Temperature for soft confidence scaling
        self.momentum = momentum # EMA momentum for stability
        
        # Confidence Scores (Vector of size [Num_Classes])
        # 0.0 = Totally unreliable (Noise), 1.0 = Highly reliable (Perfect alignment)
        self.register_buffer('class_confidence_v', torch.zeros(num_classes))
        self.register_buffer('class_confidence_r', torch.zeros(num_classes))
        
        # Raw Similarity stats for debugging
        self.register_buffer('raw_sim_v2r', torch.zeros(num_classes))
        self.register_buffer('raw_sim_r2v', torch.zeros(num_classes))
        
        # State
        self.current_epoch = 0
        self.is_ready = False
        
        # Detailed Diagnostics Storage
        self.diag_stats = {}
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    @torch.no_grad()
    def prepare_rectified_prototypes(self, prototypes_r, prototypes_v, diffusion_bridge):
        """
        Called at the beginning of Phase 2 epochs.
        Generates pseudo features and calculates reliability scores for the memory bank.
        """
        device = prototypes_r.device
        all_labels = torch.arange(self.num_classes, device=device)
        
        # =========================================================
        # 1. Generate Pseudo Prototypes via Diffusion
        # =========================================================
        # Input: Real IR Protos -> Output: Pseudo RGB Protos
        _, pseudo_v, _, unc_r2v = diffusion_bridge.generate(
            f_r=prototypes_r,
            labels_r=all_labels
        )
        
        # Input: Real RGB Protos -> Output: Pseudo IR Protos
        pseudo_r, _, unc_v2r, _ = diffusion_bridge.generate(
            f_v=prototypes_v,
            labels_v=all_labels
        )
        
        # Safety Check
        if pseudo_v is None or pseudo_r is None:
            self._handle_generation_failure()
            return

        # Normalize features
        pseudo_v = F.normalize(pseudo_v, dim=1) # Pseudo RGB
        pseudo_r = F.normalize(pseudo_r, dim=1) # Pseudo IR
        real_v = F.normalize(prototypes_v, dim=1) # Real RGB Memory
        real_r = F.normalize(prototypes_r, dim=1) # Real IR Memory
        
        # =========================================================
        # 2. Rectification: Verify against Memory Bank
        # =========================================================
        # Calculate Cosine Similarity between Generated and Real Memory
        # Ideally, Pseudo_RGB should be very close to Real_RGB for the same class
        
        # Sim(Pseudo RGB, Real RGB)
        sim_r2v = F.cosine_similarity(pseudo_v, real_v, dim=1)
        # Sim(Pseudo IR, Real IR)
        sim_v2r = F.cosine_similarity(pseudo_r, real_r, dim=1)
        
        # Store raw similarities for debugging
        self.raw_sim_r2v = sim_r2v
        self.raw_sim_v2r = sim_v2r
        
        # =========================================================
        # 3. Compute Confidence Scores (Soft Gating)
        # =========================================================
        # Formula: Sigmoid( (Similarity - Threshold) / Temperature )
        # This creates a soft gate: High similarity -> Confidence ~ 1.0, Low -> 0.0
        
        conf_v = torch.sigmoid((sim_r2v - self.rectification_threshold) / self.temp)
        conf_r = torch.sigmoid((sim_v2r - self.rectification_threshold) / self.temp)
        
        # Filter out absolute noise (Hard Cutoff for safety)
        # If similarity is extremely low (e.g. < 0.05), force confidence to 0
        hard_cutoff = 0.05
        conf_v[sim_r2v < hard_cutoff] = 0.0
        conf_r[sim_v2r < hard_cutoff] = 0.0
        
        # =========================================================
        # 4. Uncertainty Penalty (Optional but Recommended)
        # =========================================================
        # If diffusion says "I am uncertain", reduce confidence
        if unc_r2v is not None:
             # unc typically ~0.8 to 1.2. Map to penalty factor.
             uncertainty_penalty = torch.exp(-2.0 * (unc_r2v - 0.5).clamp(min=0))
             conf_v = conf_v * uncertainty_penalty
             
        if unc_v2r is not None:
             uncertainty_penalty = torch.exp(-2.0 * (unc_v2r - 0.5).clamp(min=0))
             conf_r = conf_r * uncertainty_penalty

        # =========================================================
        # 5. Update State (EMA or Direct Update)
        # =========================================================
        if self.is_ready:
            # EMA Update to prevent oscillation
            self.class_confidence_v = self.momentum * self.class_confidence_v + (1 - self.momentum) * conf_v
            self.class_confidence_r = self.momentum * self.class_confidence_r + (1 - self.momentum) * conf_r
        else:
            # First initialization
            self.class_confidence_v = conf_v
            self.class_confidence_r = conf_r
            self.is_ready = True
            
        # Update Debug Stats
        self._update_diagnostics(sim_v2r, sim_r2v, conf_r, conf_v)
        
        # Console Log for sanity check
        print(f"[PRUD] Rectified Prototypes Updated.")
        print(f"       Avg Sim: V->R {sim_v2r.mean():.4f} | R->V {sim_r2v.mean():.4f}")
        print(f"       Avg Conf: V->R {self.class_confidence_r.mean():.4f} | R->V {self.class_confidence_v.mean():.4f}")

    def get_distillation_weights(self, rgb_ids, ir_ids):
        """
        Retrieves confidence weights for the current batch.
        Args:
            rgb_ids: Batch RGB labels (shape: B)
            ir_ids: Batch IR labels (shape: B)
        Returns:
            weight_v: Confidence for RGB samples (based on IR->V rectification)
            weight_r: Confidence for IR samples (based on V->R rectification)
        """
        if not self.is_ready:
            # Return zeros if not ready (Safe Mode)
            return torch.zeros_like(rgb_ids, dtype=torch.float), torch.zeros_like(ir_ids, dtype=torch.float)
        
        # Lookup confidence from the buffer
        # For RGB samples, we trust them if the R->V generation matches the RGB prototype
        weight_v = self.class_confidence_v[rgb_ids]
        
        # For IR samples, we trust them if the V->R generation matches the IR prototype
        weight_r = self.class_confidence_r[ir_ids]
        
        return weight_v.detach(), weight_r.detach()

    def _handle_generation_failure(self):
        print("⚠️ [PRUD] Warning: Diffusion generation failed/returned None. PRUD is disabled for this epoch.")
        self.is_ready = False
        self.class_confidence_v.fill_(0.0)
        self.class_confidence_r.fill_(0.0)

    def _update_diagnostics(self, sim_v2r, sim_r2v, conf_r, conf_v):
        """Updates internal dictionary for file logging"""
        self.diag_stats = {
            'sim_v2r_mean': sim_v2r.mean().item(),
            'sim_r2v_mean': sim_r2v.mean().item(),
            'sim_v2r_max': sim_v2r.max().item(),
            'sim_r2v_max': sim_r2v.max().item(),
            'conf_r_mean': conf_r.mean().item(), # IR Confidence
            'conf_v_mean': conf_v.mean().item(), # RGB Confidence
            'reliable_classes_v': (conf_v > 0.5).sum().item(),
            'reliable_classes_r': (conf_r > 0.5).sum().item(),
            'zero_conf_classes': ((conf_v < 0.01) | (conf_r < 0.01)).sum().item()
        }

    def get_detailed_diagnostics(self):
        """
        Returns a formatted string for the logger.
        Shows detailed distribution of similarities and confidences.
        """
        if not self.is_ready:
            return "🛡️ PRUD: Not Ready (Prototypes not rectified yet)"
        
        s = self.diag_stats
        lines = []
        lines.append(f"🛡️ PRUD (Prototype Rectification) Diagnostics:")
        lines.append(f"   ├─ Status: {'✓ Active' if self.is_ready else '⚠️ Inactive'}")
        lines.append(f"   ├─ Parameters: Threshold={self.rectification_threshold}, Temp={self.temp}")
        lines.append(f"   │")
        lines.append(f"   ├─ Prototype Consistency (Sim(Fake, Real)):")
        lines.append(f"   │   ├─ V→Pseudo_R vs Real_R: Avg {s.get('sim_v2r_mean', 0):.4f} | Max {s.get('sim_v2r_max', 0):.4f}")
        lines.append(f"   │   └─ R→Pseudo_V vs Real_V: Avg {s.get('sim_r2v_mean', 0):.4f} | Max {s.get('sim_r2v_max', 0):.4f}")
        lines.append(f"   │")
        lines.append(f"   ├─ Class Confidence Scores (Weights for Distillation):")
        lines.append(f"   │   ├─ IR Confidence (from V→R): Avg {s.get('conf_r_mean', 0):.4f}")
        lines.append(f"   │   └─ RGB Confidence (from R→V): Avg {s.get('conf_v_mean', 0):.4f}")
        lines.append(f"   │")
        lines.append(f"   └─ Reliability Distribution (Total {self.num_classes} Classes):")
        lines.append(f"       ├─ Highly Reliable (>0.5): Vis {s.get('reliable_classes_v', 0)} | IR {s.get('reliable_classes_r', 0)}")
        lines.append(f"       └─ Totally Rejected (<0.01): {s.get('zero_conf_classes', 0)} classes (Noise)")
        
        return '\n'.join(lines)