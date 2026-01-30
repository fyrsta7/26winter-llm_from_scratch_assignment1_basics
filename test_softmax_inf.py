#!/usr/bin/env python3
"""
测试softmax处理-inf的行为
"""
import torch
import torch.nn.functional as F
from cs336_basics.model import softmax

# 测试1: 包含-inf的tensor
print("=" * 80)
print("测试1: 包含-inf的tensor")
print("=" * 80)

x = torch.tensor([[1.0, 2.0, float('-inf'), 3.0],
                  [float('-inf'), float('-inf'), 1.0, 2.0]])

print("Input:")
print(x)

our_result = softmax(x, dim=-1)
pytorch_result = F.softmax(x, dim=-1)

print("\nOur softmax:")
print(our_result)

print("\nPyTorch F.softmax:")
print(pytorch_result)

print("\nDifference:")
print(torch.abs(our_result - pytorch_result))

print(f"\nAll close? {torch.allclose(our_result, pytorch_result, atol=1e-6, equal_nan=True)}")

# 测试2: 全部是-inf的行
print("\n" + "=" * 80)
print("测试2: 全部是-inf的行")
print("=" * 80)

x2 = torch.tensor([[1.0, 2.0, 3.0],
                   [float('-inf'), float('-inf'), float('-inf')]])

print("Input:")
print(x2)

our_result2 = softmax(x2, dim=-1)
pytorch_result2 = F.softmax(x2, dim=-1)

print("\nOur softmax:")
print(our_result2)

print("\nPyTorch F.softmax:")
print(pytorch_result2)

print("\nDifference:")
print(torch.abs(our_result2 - pytorch_result2))

print(f"\nAll close (with nan=True)? {torch.allclose(our_result2, pytorch_result2, atol=1e-6, equal_nan=True)}")

# 测试3: 使用实际的attention scores
print("\n" + "=" * 80)
print("测试3: 实际attention测试场景")
print("=" * 80)

# 使用与测试相同的参数
batch_size = 4
n_queries = 12
n_keys = 16
d_model = 64

torch.manual_seed(1)
q = torch.randn(batch_size, n_queries, d_model)
torch.manual_seed(2)
k = torch.randn(batch_size, n_keys, d_model)
torch.manual_seed(5)
mask = torch.randn(batch_size, n_queries, n_keys) > 0.5

# 计算attention scores
scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model, dtype=q.dtype))

# 应用mask
scores_masked = scores.masked_fill(mask, float('-inf'))

print(f"Scores shape: {scores.shape}")
print(f"Number of -inf values: {torch.isinf(scores_masked).sum()}")
print(f"Scores sample (first query, first 5 keys):")
print(scores_masked[0, 0, :5])

# 比较softmax
our_attn = softmax(scores_masked, dim=-1)
pytorch_attn = F.softmax(scores_masked, dim=-1)

print(f"\nOur softmax (first query, first 5 weights):")
print(our_attn[0, 0, :5])

print(f"\nPyTorch softmax (first query, first 5 weights):")
print(pytorch_attn[0, 0, :5])

diff = torch.abs(our_attn - pytorch_attn)
max_diff = torch.max(diff[~torch.isnan(diff)]).item() if not torch.isnan(diff).all() else 0.0

print(f"\nMax difference (excluding NaN): {max_diff}")
print(f"All close? {torch.allclose(our_attn, pytorch_attn, atol=1e-6, equal_nan=True)}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("如果我们的softmax与PyTorch的F.softmax在所有情况下都一致，")
print("说明我们的softmax实现是正确的，包括对-inf的处理。")
