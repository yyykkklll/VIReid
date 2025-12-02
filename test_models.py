"""
test_models.py - 模型单元测试脚本

用法:
    python test_models.py --test pfmgcd
    python test_models.py --test memory_bank
    python test_models.py --test graph_propagation
    python test_models.py --test all
"""

import sys
import os
import argparse
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_pfmgcd():
    """测试PF-MGCD主模型"""
    from models.pfmgcd_model import PF_MGCD
    
    print("="*60)
    print("PF-MGCD 模型测试")
    print("="*60)
    
    model = PF_MGCD(
        num_parts=6,
        num_identities=395,
        feature_dim=256,
        pretrained=False
    )
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 3, 288, 144)
    labels = torch.randint(0, 395, (batch_size,))
    
    print(f"输入shape: {x.shape}")
    
    model.train()
    outputs = model(x, labels=labels)
    
    print(f"\n输出keys: {outputs.keys()}")
    print(f"id_features[0] shape: {outputs['id_features'][0].shape}")
    print(f"id_logits[0] shape: {outputs['id_logits'][0].shape}")
    
    model.eval()
    feat = model.extract_features(x, pool_parts=True)
    print(f"提取特征shape: {feat.shape}")
    
    print("\n✅ PF-MGCD测试通过!\n")


def test_memory_bank():
    """测试记忆库"""
    from models.memory_bank import MultiPartMemoryBank
    
    print("="*60)
    print("Memory Bank 测试")
    print("="*60)
    
    num_parts = 6
    num_identities = 100
    feature_dim = 256
    batch_size = 8
    
    memory_bank = MultiPartMemoryBank(
        num_parts=num_parts,
        num_identities=num_identities,
        feature_dim=feature_dim,
        momentum=0.9
    )
    
    print(f"Memory shape: {memory_bank.memory.shape}")
    
    part_features = [torch.randn(batch_size, feature_dim) for _ in range(num_parts)]
    labels = torch.randint(0, num_identities, (batch_size,))
    
    print(f"Labels: {labels}")
    
    # 初始化
    memory_bank.initialize_memory(part_features, labels)
    print(f"已初始化: {memory_bank.initialized.sum().item()}/{num_identities}")
    
    # 更新
    memory_bank.update_memory(part_features, labels)
    print("记忆库更新成功")
    
    print("\n✅ Memory Bank测试通过!\n")


def test_graph_propagation():
    """测试图传播"""
    from models.graph_propagation import AdaptiveGraphPropagation
    from models.memory_bank import MultiPartMemoryBank
    
    print("="*60)
    print("Graph Propagation 测试")
    print("="*60)
    
    batch_size = 8
    num_parts = 6
    num_classes = 100
    feature_dim = 256
    
    part_features = [torch.randn(batch_size, feature_dim) for _ in range(num_parts)]
    memory_bank = MultiPartMemoryBank(num_parts, num_classes, feature_dim)
    
    graph_prop = AdaptiveGraphPropagation(
        temperature=3.0,
        use_entropy_weight=True,
        scale=30.0
    )
    
    soft_labels, similarities, entropy_weights = graph_prop(part_features, memory_bank)
    
    print(f"软标签shape: {soft_labels[0].shape}")
    print(f"概率和: {soft_labels[0][0].sum().item():.4f}")
    
    if entropy_weights:
        print(f"熵权重均值: {entropy_weights[0].mean().item():.4f}")
    
    print("\n✅ Graph Propagation测试通过!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all',
                        choices=['pfmgcd', 'memory_bank', 'graph_propagation', 'all'])
    args = parser.parse_args()
    
    if args.test == 'pfmgcd' or args.test == 'all':
        test_pfmgcd()
    
    if args.test == 'memory_bank' or args.test == 'all':
        test_memory_bank()
    
    if args.test == 'graph_propagation' or args.test == 'all':
        test_graph_propagation()
    
    print("="*60)
    print("🎉 所有测试完成!")
    print("="*60)


if __name__ == '__main__':
    main()
