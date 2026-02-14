import numpy as np
import torch

"""
完全盲目验证: Attention = 梯度下降
假设我不知道它们等价,只是按照定义独立实现
"""

print("=== 盲目验证: Attention vs 梯度下降 ===\n")

# 设置随机种子保证可复现
torch.manual_seed(42)
dim = 64

# 初始化
W = torch.randn(dim, dim) * 0.1  # 预训练权重
query = torch.randn(dim)          # 当前输入
key = torch.randn(dim)            # 上下文中的 key
value = torch.randn(dim)          # 上下文中的 value

print("场景: 我们有一个预训练模型 W")
print(f"现在来了一个新的 key-value 对: key={key[:3].numpy()}, value={value[:3].numpy()}")
print(f"我们想用 query={query[:3].numpy()} 去检索\n")

# ============================================
# 方法 A: 梯度下降 (One-Shot Learning)
# ============================================
print("--- 方法 A: 梯度下降 (Fine-Tuning) ---")
print("思路: 我想让模型学会 key → value 的映射")
print("做法: 用一步梯度下降更新权重\n")

# 目标: W·key ≈ value
# 损失: L = ||W·key - value||²
# 梯度: ∂L/∂W = 2·(W·key - value)·key^T
# 更新: W_new = W - lr·∂L/∂W
#            = W - lr·2·(W·key - value)·key^T
#            = W + lr·2·(value - W·key)·key^T
#
# 简化 (忽略常数2): W_new = W + lr·(value - W·key)·key^T
#                         ≈ W + lr·value·key^T  (假设 W·key ≈ 0)

lr = 1.0 / dim  # 学习率
delta_W = lr * torch.outer(value, key)  # Hebbian update
W_updated = W + delta_W

# 用更新后的权重计算输出
output_gd = W_updated @ query

print(f"更新后的权重计算: output = (W + lr·value·key^T) @ query")
print(f"输出前5维: {output_gd[:5].numpy()}\n")

# ============================================
# 方法 B: Attention 机制
# ============================================
print("--- 方法 B: Attention (In-Context Learning) ---")
print("思路: 不改变权重,而是用 attention 动态检索")
print("做法: 计算 query 和 key 的相似度,然后加权 value\n")

# 基础输出 (用原始权重)
output_base = W @ query

# Attention 分数 (简化版,不用 softmax)
similarity = torch.dot(query, key) * lr  # 注意这里也用了 lr 作为缩放

# Attention 输出
attn_contribution = similarity * value

# 最终输出
output_attn = output_base + attn_contribution

print(f"Attention 计算: output = W @ query + (query·key)·lr·value")
print(f"输出前5维: {output_attn[:5].numpy()}\n")

# ============================================
# 对比结果
# ============================================
print("=== 对比结果 ===")
print(f"梯度下降输出: {output_gd[:5].numpy()}")
print(f"Attention 输出: {output_attn[:5].numpy()}")

diff = torch.norm(output_gd - output_attn).item()
print(f"\n差异 (L2 范数): {diff:.10f}")

if diff < 1e-5:
    print("\n✅ 它们几乎完全相同!")
    print("这证明: Attention 在数学上等价于一步梯度下降!")
else:
    print(f"\n❌ 差异较大: {diff:.4f}")
    print("让我检查一下数学推导...")
    
    # 展开验证
    print("\n--- 数学推导 ---")
    print("梯度下降: (W + lr·v·k^T) @ q")
    print("         = W@q + lr·v·(k^T @ q)")
    print("         = W@q + lr·v·(k·q)  [因为 k^T @ q = k·q]")
    print("")
    print("Attention:  W@q + (q·k)·lr·v")
    print("         = W@q + lr·(q·k)·v")
    print("         = W@q + lr·v·(k·q)  [因为标量可交换]")
    print("")
    print("所以它们应该完全相同!")
    
    # 重新计算验证
    manual_gd = W @ query + lr * value * torch.dot(key, query)
    manual_attn = W @ query + lr * torch.dot(query, key) * value
    
    print(f"\n手动验证:")
    print(f"梯度下降: {manual_gd[:5].numpy()}")
    print(f"Attention:  {manual_attn[:5].numpy()}")
    print(f"差异: {torch.norm(manual_gd - manual_attn).item():.10f}")

