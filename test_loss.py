# test_loss.py
from models.loss import TotalLoss
import torch

# 测试不带use_adaptive_weight
loss_fn1 = TotalLoss(num_parts=6, use_adaptive_weight=False)
print("✓ TotalLoss without adaptive weight: OK")

# 测试带use_adaptive_weight
loss_fn2 = TotalLoss(num_parts=6, use_adaptive_weight=True)
print("✓ TotalLoss with adaptive weight: OK")

# 模拟输入
outputs = {
    'id_features': [torch.randn(8, 256) for _ in range(6)],
    'id_logits': [torch.randn(8, 206) for _ in range(6)],
    'mod_logits': [torch.randn(8, 2) for _ in range(6)],
    'soft_labels': [torch.randn(8, 206) for _ in range(6)]
}
labels = torch.randint(0, 206, (8,))
modality_labels = torch.randint(0, 2, (8,))

# 前向传播
loss, loss_dict = loss_fn1(outputs, labels, modality_labels, current_epoch=10)
print(f"✓ Loss computation: {loss.item():.4f}")
print(f"✓ Loss dict keys: {list(loss_dict.keys())}")

print("\nAll tests passed! ✓")
