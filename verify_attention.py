#!/usr/bin/env python3
"""
验证attention实现是否正确
对比我们的实现和PyTorch标准实现
"""
import torch
import torch.nn.functional as F
import numpy as np

# 使用与测试相同的fixture参数
batch_size = 4
n_queries = 12
n_keys = 16
n_heads = 4
d_head = 16
d_model = n_heads * d_head  # 64

# 使用与测试相同的随机种子生成数据
torch.manual_seed(1)
q = torch.randn(batch_size, n_queries, d_model)

torch.manual_seed(2)
k = torch.randn(batch_size, n_keys, d_model)

torch.manual_seed(3)
v = torch.randn(batch_size, n_keys, d_model)

torch.manual_seed(5)
mask = torch.randn(batch_size, n_queries, n_keys) > 0.5

print("=" * 80)
print("测试1: 使用我们的实现")
print("=" * 80)

from cs336_basics.model import scaled_dot_product_attention
output_ours = scaled_dot_product_attention(q, k, v, mask)

print(f"Q shape: {q.shape}")
print(f"K shape: {k.shape}")
print(f"V shape: {v.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Output shape: {output_ours.shape}")
print(f"\nOur output (first 5 values): {output_ours[0, 0, :5]}")

print("\n" + "=" * 80)
print("测试2: 使用PyTorch标准实现")
print("=" * 80)

# PyTorch标准实现
d_k_val = q.shape[-1]
scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k_val, dtype=q.dtype))
if mask is not None:
    scores = scores.masked_fill(mask, float('-inf'))
attn_weights = F.softmax(scores, dim=-1)
output_pytorch = torch.matmul(attn_weights, v)

print(f"PyTorch output (first 5 values): {output_pytorch[0, 0, :5]}")

print("\n" + "=" * 80)
print("测试3: 对比差异")
print("=" * 80)

diff = torch.abs(output_ours - output_pytorch)
max_diff = torch.max(diff).item()
mean_diff = torch.mean(diff).item()

print(f"Max difference: {max_diff}")
print(f"Mean difference: {mean_diff}")
print(f"All close (atol=1e-6)? {torch.allclose(output_ours, output_pytorch, atol=1e-6)}")

print("\n" + "=" * 80)
print("测试4: 检查快照数据")
print("=" * 80)

# 尝试加载快照
snapshot_path = "tests/_snapshots/test_scaled_dot_product_attention.npz"
try:
    snapshot = np.load(snapshot_path)
    expected = torch.from_numpy(snapshot['array'])
    
    print(f"Expected output (first 5 values): {expected[0, 0, :5]}")
    
    diff_expected = torch.abs(output_ours - expected)
    max_diff_expected = torch.max(diff_expected).item()
    mean_diff_expected = torch.mean(diff_expected).item()
    
    print(f"\nDifference with snapshot:")
    print(f"  Max difference: {max_diff_expected}")
    print(f"  Mean difference: {mean_diff_expected}")
    print(f"  All close (atol=1e-6)? {torch.allclose(output_ours, expected, atol=1e-6)}")
    
    # 对比PyTorch实现和快照
    diff_pytorch_expected = torch.abs(output_pytorch - expected)
    max_diff_pytorch_expected = torch.max(diff_pytorch_expected).item()
    mean_diff_pytorch_expected = torch.mean(diff_pytorch_expected).item()
    
    print(f"\nPyTorch vs snapshot:")
    print(f"  Max difference: {max_diff_pytorch_expected}")
    print(f"  Mean difference: {mean_diff_pytorch_expected}")
    print(f"  All close (atol=1e-6)? {torch.allclose(output_pytorch, expected, atol=1e-6)}")
    
except Exception as e:
    print(f"无法加载快照: {e}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("如果我们的实现与PyTorch标准实现一致（误差<1e-6），但与快照不匹配，")
print("说明快照可能是用不同的实现或测试数据生成的。")
print("这种情况下，我们的实现是正确的。")
