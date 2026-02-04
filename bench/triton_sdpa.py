import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver
import math
from utils import LaunchParam

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Ground Truth: SDPA (PyTorch 参考实现)
# ============================================================================

@torch.compile
def torch_compile_sdpa(query, key, value, attn_mask) -> torch.Tensor:
    """PyTorch SDPA 参考实现 (配合 torch.compile)"""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


def torch_native_sdpa(query, key, value, attn_mask) -> torch.Tensor:
    """PyTorch SDPA 原生实现 (无 torch.compile)"""
    import pdb; pdb.set_trace()
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


# ============================================================================
# Triton Kernel v1 (TODO: 待实现)
# ============================================================================

@triton.jit
def sdpa_kernel_v1(
    query_ptr, key_ptr, value_ptr, mask_ptr, output_ptr,
    attn_ptr,
    B, seq_len_q, seq_len_kv, D,
    scale_factor,
    q_stride,
    k_stride,
    v_stride,
    attn_stride,
    out_stride,
    
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    参考算法:
    1. 加载 Q, K, V 块
    2. 计算 Q @ K^T * scale
    3. Online softmax
    4. 累加 attention @ V
    5. 写回输出
    """
    bid = tl.program_id(0)

    # batch 
    if bid < B:
        q_start = query_ptr + bid * seq_len_q * q_stride
        k_start = key_ptr + bid * seq_len_kv * k_stride
        v_start = value_ptr + bid * seq_len_kv * v_stride
        attn_start = attn_ptr + bid * seq_len_q * seq_len_kv
        out_start = output_ptr + bid * seq_len_kv, v_stride

        # target 算 softmax buf
        # ----- | ---- |
        # ----- | ---- | 

        for m in range(0, seq_len_q, BLOCK_M):

            _max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            _exp_sum = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

            offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q
            offs_d = tl.arange(0, BLOCK_D)

            for n in range(0, seq_len_kv, BLOCK_N):
                # Q 偏移计算
                q_ptrs = q_start + offs_m[:, None] + offs_d[None, :]

                # K 转置偏移
                offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len_kv
                k_ptrs = k_start + offs_n[None, :] * k_stride + offs_d[None, :]

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for _ in range(0, D, BLOCK_D):
                    q = tl.load(q_ptrs, mask=mask_m)
                    k = tl.load(k_ptrs, mask=mask_n)
                    acc += tl.dot(q, k)
                    
                    q_ptrs += BLOCK_D
                    k_ptrs += BLOCK_D * k_stride

                acc = acc * scale_factor
                offs_attn = offs_m[:, None] + offs_n[None, :]

                attn_fuse_mask = tl.load(mask_ptr+ offs_attn, mask=mask_m&mask_n, other=False)
                # acc 是 softmax 输入的 [M, N] 一小块。考虑外部输入的 mask，一起跑 online-softmax
                tl.store(attn_start+offs_attn, acc, mask=attn_fuse_mask, other=tl.float32('-inf'))

                block_mask = tl.max(acc, mask=attn_fuse_mask, axis=-1)
                new_max = tl.maxisum(_max, block_mask)
                _exp_sum = _exp_sum * tl.exp(_max - new_max) + tl.exp(acc - new_max, mask=attn_fuse_mask) 
                _max = new_max
            
            for _ in range(0, seq_len_kv, BLOCK_N):
                # 算 attn score
                offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len_kv
                offs_attn = offs_m[:, None] + offs_n[None, :]
                mask_attn = mask_m & mask_n
                attn = tl.load(attn_start+offs_attn, mask=mask_attn, other=tl.float32('-inf'))
                attn_score = tl.exp(attn-_max) / _exp_sum

                tl.store(attn_start+offs_attn, attn_score, mask=mask_attn)
            
        # attn_score @ V
        # [seq_len_q, seq_len_kv] @ [seq_len_kv, D]

        for m in range(0, seq_len_q, BLOCK_M):
            offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q
            
            for n in range(0, D, BLOCK_N):
                offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
                attn_ptrs = attn_start + offs_m[:, None] + offs_d[None, :]

                mask_n = offs_n < D
                v_ptrs = v_start + offs_n[None, :] * v_stride + offs_d[None, :]

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for _ in range(0, D, BLOCK_D):
                    q = tl.load(attn_ptrs, mask=mask_m)
                    k = tl.load(v_ptrs, mask=mask_n)
                    acc += tl.dot(q, k)
                    
                    attn_ptrs += BLOCK_D
                    v_ptrs += BLOCK_D * v_stride
                
                tl.store(out_start+offs_m[:, None] + offs_n[None, :], mask=mask_m&mask_n)


def v1(query, key, value, attn_mask):
    """
    TODO: SDPA v1 实现
    
    输入形状:
        query: (batch_size, num_heads_q, seq_len_q, head_dim)
        key:   (batch_size, num_heads_kv, seq_len_kv, head_dim)
        value: (batch_size, num_heads_kv, seq_len_kv, head_dim)
        attn_mask: (seq_len_q, seq_len_kv)
    输出形状:
        output: (batch_size, num_heads_q, seq_len_q, head_dim)
    """
    batch_size, num_heads_q, seq_len_q, head_dim = query.shape
    _, num_heads_kv, seq_len_kv, _ = key.shape
    
    # 默认 scale
    scale = 1.0 / math.sqrt(head_dim)
    
    # 分配输出
    output = torch.empty_like(query)
    
    # TODO: 启动 kernel
    # sdpa_kernel_v1[grid](...)
    
    raise NotImplementedError("SDPA v1 kernel not implemented yet")
    
    return output


# ============================================================================
# Precision Check
# ============================================================================

def precision_check_all():
    """精度对比测试"""
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    num_heads = 8
    seq_len = 512
    head_dim = 64
    
    # 生成测试输入
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    
    # 定义测试场景: (attn_mask, desc)
    test_cases = [
        # (torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool), "bool mask (all True)"),
        (torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool).tril(diagonal=0), "bool mask (causal)"),
    ]
    
    for attn_mask, desc in test_cases:
        print(f"\nTest case: {desc}")
        
        # PyTorch 参考输出
        try:
            output_torch = torch_native_sdpa(query, key, value, attn_mask)
            print(f"  PyTorch native output sample: {output_torch[0, 0, 0, :5]}")
        except Exception as e:
            print(f"  PyTorch native error: {e}")
            continue
        
        # 测试各个版本
        for v_name, v_fn in [
            ("torch_compile", torch_compile_sdpa),
            ("v1", v1),
        ]:
            try:
                q = query.clone()
                k = key.clone()
                v = value.clone()
                m = attn_mask.clone()
                output_triton = v_fn(q, k, v, m)
                max_diff = torch.max(torch.abs(output_torch - output_triton))
                print(f"  {v_name} max diff vs torch: {max_diff:.6f}")
            except NotImplementedError as e:
                print(f"  {v_name}: {e}")


# ============================================================================
# Benchmark
# ============================================================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],  # x 轴变量
        x_vals=[128 * i for i in range(2, 64, 4)],  # 序列长度变化范围
        line_arg='provider',  # 不同线条对应不同实现
        line_vals=['torch', 'tc', 'v1'],
        line_names=["PyTorch", "TorchCompile", "Triton-v1"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="TFLOPS",  # 注意：SDPA 用 TFLOPS 更合适
        plot_name="sdpa-performance",
        args={'batch_size': 2, 'num_heads': 8, 'head_dim': 64},
    ))
def benchmark(batch_size, num_heads, seq_len, head_dim, provider):
    """SDPA 性能基准测试"""
    # 生成输入
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    
    # 计算 FLOPs (近似)
    # Q @ K^T: 2 * batch * heads * seq_len^2 * head_dim
    # Softmax: 5 * batch * heads * seq_len^2 (approx)
    # Attn @ V: 2 * batch * heads * seq_len^2 * head_dim
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    # 生成 causal mask
    attn_mask = torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool).tril(diagonal=0)
    
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_native_sdpa(query, key, value, attn_mask))
    elif provider == 'tc':
        ms = triton.testing.do_bench(lambda: torch_compile_sdpa(query, key, value, attn_mask))
    elif provider == 'v1':
        try:
            ms = triton.testing.do_bench(lambda: v1(query, key, value, attn_mask))
        except NotImplementedError:
            return 0  # 未实现时返回 0
    
    # 转换为 TFLOPS
    tflops = lambda ms: flops * 1e-12 / (ms * 1e-3)
    return tflops(ms)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Running precision check...")
    precision_check_all()
    
    print("\nRunning benchmark...")
    benchmark.run(save_path=os.path.dirname(__file__), print_data=True)
