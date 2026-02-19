import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver
import math
from utils import LaunchParam, calculate_settings
import pdb
import tritonparse.structured_logging
import tritonparse.parse.utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize logging with full tracing options
tritonparse.structured_logging.init(
    "./logs/",
    enable_trace_launch=True,                 # Capture kernel launch events (enables torch.compile tracing automatically)
    enable_more_tensor_information=True,      # Optional: collect tensor statistics (min/max/mean/std)
)

# ============================================================================
# Triton Kernel v3 (FlashAttention with 2D grid parallelization)
# ============================================================================

@triton.jit
def sdpa_kernel_v3(
    query_ptr, key_ptr, value_ptr, mask_ptr, output_ptr,
    B, seq_len_q, seq_len_kv, head_dim,
    scale_factor,
    
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention with 2D grid:
    - grid.x = batch * heads (不同的 head 并行)
    - grid.y = seq_len_q / BLOCK_M (不同的 query block 并行)
    
    这样可以将大量的 warps/blocks 分配到 GPU 上，提高 SM 利用率
    """
    bid = tl.program_id(0)  # batch * head index
    pid_m = tl.program_id(1)  # query block index
    
    if bid < B:
        q_start = query_ptr + bid * seq_len_q * head_dim
        k_start = key_ptr + bid * seq_len_kv * head_dim
        v_start = value_ptr + bid * seq_len_kv * head_dim
        mask_start = mask_ptr
        out_start = output_ptr + bid * seq_len_q * head_dim

        # 每个 block 处理一个 query tile
        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q

        # Online-softmax 统计量
        m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        
        # 输出累加器 [BLOCK_M, BLOCK_D]
        acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        
        # 遍历 seq_len_kv 维度
        for n in range(0, seq_len_kv, BLOCK_N):
            offs_n = n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_kv

            # 计算 Q @ K^T: [BLOCK_M, BLOCK_N]
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            for d in range(0, head_dim, BLOCK_D):
                offs_d = d + tl.arange(0, BLOCK_D)
                mask_d = offs_d < head_dim
                
                q_ptrs = q_start + offs_m[:,None] * head_dim + offs_d[None, :]
                q = tl.load(q_ptrs, mask=mask_m[:,None] & mask_d[None,:], other=0.0)
                
                k_ptrs = k_start + offs_n[:,None] * head_dim + offs_d[None, :]
                k = tl.load(k_ptrs, mask=mask_n[:,None] & mask_d[None,:], other=0.0)
                
                qk += tl.dot(q, tl.trans(k)).to(tl.float32)

            # 应用 scale 和 mask
            qk_scaled = qk * scale_factor
            
            # 加载外部 mask
            offs_mask = offs_m[:,None] * seq_len_kv + offs_n[None, :]
            attn_mask = tl.load(mask_start + offs_mask, mask=mask_m[:,None] & mask_n[None,:], other=False)
            
            # 优化：检查 mask 是否全为 False（如下三角矩阵的右上角区域）
            # 如果是，则跳过该 block 的计算，避免不必要的 softmax 和 V 加载
            mask_any = tl.max(attn_mask.to(tl.int32))
            if mask_any > 0:
                qk_masked = tl.where(attn_mask, qk_scaled, float('-inf'))

                # ===== Online softmax 更新 =====
                m_curr = tl.max(qk_masked, axis=1)
                m_new = tl.maximum(m_prev, m_curr)
                alpha = tl.exp(m_prev - m_new)
                p = tl.exp(qk_masked - m_new[:, None])
                l_curr = tl.sum(p, axis=1)
                l_new = alpha * l_prev + l_curr
                
                # 修正输出累加器
                acc_o = acc_o * alpha[:, None]
                
                # 加载 V 并累加
                for d in range(0, head_dim, BLOCK_D):
                    offs_d = d + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < head_dim
                    
                    v_ptrs = v_start + offs_n[:,None] * head_dim + offs_d[None, :]
                    v = tl.load(v_ptrs, mask=mask_n[:,None] & mask_d[None,:], other=0.0)
                    acc_o += tl.dot(p, v)

                m_prev = m_new
                l_prev = l_new

        # 最终归一化并写回
        acc_o = acc_o / l_prev[:, None]
        
        for d in range(0, head_dim, BLOCK_D):
            offs_d = d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < head_dim
            out_ptrs = out_start + offs_m[:,None] * head_dim + offs_d[None, :]
            tl.store(out_ptrs, acc_o[:, :], mask=mask_m[:,None] & mask_d[None,:])


def v3(query, key, value, attn_mask):
    """
    SDPA v3 实现 - FlashAttention with 2D grid parallelization
    """
    batch_size, num_heads_q, seq_len_q, head_dim = query.shape
    _, num_heads_kv, seq_len_kv, _ = key.shape
    
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_mask = attn_mask.contiguous()
    
    scale = 1.0 / math.sqrt(head_dim)
    
    query_reshaped = query.view(batch_size * num_heads_q, seq_len_q, head_dim)
    key_reshaped = key.view(batch_size * num_heads_kv, seq_len_kv, head_dim)
    value_reshaped = value.view(batch_size * num_heads_kv, seq_len_kv, head_dim)
    
    output = torch.empty_like(query_reshaped)
    
    B = batch_size * num_heads_q
    
    # 使用 calculate_settings 计算 BLOCK_D 和 num_warps
    BLOCK_D, _ = calculate_settings(head_dim)
    
    # BLOCK_M 和 BLOCK_N 的选择策略：
    # 1. 小序列：用较小的 BLOCK 增加并行度
    # 2. 大序列：用较大的 BLOCK 减少 kernel 启动开销，但要考虑寄存器压力
    # 3. BLOCK_N 通常 >= BLOCK_M，这样每个 query block 可以重用加载的 KV
    
    # 计算可用的并行度
    total_queries = B * seq_len_q
    
    if seq_len_q <= 64:
        # 小序列：最大化并行度
        BLOCK_M = 64
        BLOCK_N = 64
        num_warps = 4
    elif seq_len_q <= 512:
        # 中等序列
        BLOCK_M = 64
        BLOCK_N = 128  # 更大的 BLOCK_N 减少 KV 加载次数
        num_warps = 4
    else:
        # 大序列：平衡并行度和效率
        # 如果总 queries 很多，可以用更大的 BLOCK
        if total_queries >= 4096:
            BLOCK_M = 128
            BLOCK_N = 128
            num_warps = 8
        else:
            BLOCK_M = 64
            BLOCK_N = 128
            num_warps = 4
    
    # 2D grid: (batch*heads, num_query_blocks)
    num_m_blocks = triton.cdiv(seq_len_q, BLOCK_M)
    grid = (B, num_m_blocks)
    
    sdpa_kernel_v3[grid](
        query_reshaped, key_reshaped, value_reshaped, attn_mask, output,
        B, seq_len_q, seq_len_kv, head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    
    output = output.view(batch_size, num_heads_q, seq_len_q, head_dim)
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
    q_seq_len = 128
    kv_seq_len = 256
    head_dim = 64
    
    # 生成测试输入
    query = torch.randn(batch_size, num_heads, q_seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    
    # 定义测试场景: (attn_mask, desc)
    test_cases = [
        # (torch.ones(q_seq_len, kv_seq_len, device=DEVICE, dtype=torch.bool), "bool mask (causal)"),
        (torch.ones(q_seq_len, kv_seq_len, device=DEVICE, dtype=torch.bool).tril(diagonal=0), "bool mask (causal)"),
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
        for _, v_fn in [
            # ("torch_compile", torch_native_sdpa),
            ("v3", v3),
        ]:
            q = query.clone()
            k = key.clone()
            v = value.clone()
            m = attn_mask.clone()
            v_fn(q, k, v, m)


# Parse and generate trace files
tritonparse.parse.utils.unified_parse("./logs/", out="./parsed_output")