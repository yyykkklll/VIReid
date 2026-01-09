import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphRegularizedCP(nn.Module):
    """
    Graph Regularized Consensus Propagation (GR-CP)
    Constructs affinity graphs from single-modal features and propagates
    prediction scores to enforce smoothness and consistency.
    """
    def __init__(self, k=5, alpha_max=0.3):
        super(GraphRegularizedCP, self).__init__()
        self.k = k
        self.alpha_max = alpha_max

    def build_graph(self, features):
        """
        Build normalized adjacency matrix
        Input: features (B, D)
        Output: adj (B, B)
        """
        B = features.size(0)
        # 1. Cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(features_norm, features_norm.t()) # (B, B)

        # 2. k-NN
        # Get topk values and indices
        # Ensure k is not larger than B
        k = min(self.k, B)
        topk_val, topk_idx = torch.topk(sim_matrix, k=k, dim=1)
        
        # Build sparse matrix (Scatter)
        mask = torch.zeros_like(sim_matrix)
        mask.scatter_(1, topk_idx, topk_val) # Keep only TopK similarity values
        
        # 3. Add Self-loop and Symmetric Normalization
        adj = mask + torch.eye(B).to(features.device)
        
        # Calculate degree matrix D^(-0.5)
        row_sum = adj.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # D^(-0.5) * A * D^(-0.5)
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return norm_adj

    def propagate(self, preds, adj, alpha):
        """
        Graph Propagation/Smoothing
        Input: preds (B, Class_Num), adj (B, B), alpha (float)
        """
        # Formula: P_new = (1-alpha)P + alpha * A * P
        smoothed_preds = (1 - alpha) * preds + alpha * torch.mm(adj, preds)
        return smoothed_preds

    def forward(self, features, preds, current_alpha, logger=None):
        """
        Execute Graph Regularization
        """
        if logger: logger.debug('Building Graph...')
        adj = self.build_graph(features)
        
        if logger: logger.debug('Propagating scores...')
        smoothed_preds = self.propagate(preds, adj, current_alpha)
        
        return smoothed_preds
