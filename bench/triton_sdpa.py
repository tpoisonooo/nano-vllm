import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver
import math
from utils import LaunchParam, calculate_settings
import pdb
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
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value

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
# Triton Kernel v2 (FlashAttention-style: fuse attn@V into online-softmax)
# ============================================================================

@triton.jit
def sdpa_kernel_v2(
    query_ptr, key_ptr, value_ptr, mask_ptr, output_ptr,
    B, seq_len_q, seq_len_kv, head_dim,
    scale_factor,
    
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention-style fused SDPA:
    核心思想：不需要存储完整的 attention matrix [seq_len_q, seq_len_kv]
    而是在计算每个 KV block 的 softmax 时，立即与 V 相乘并累加到输出
    
    Online-softmax 累加技巧：
    - 维护 running max (m) 和 running sum (l)
    - 当发现更大的 max 时，需要修正之前的累加结果
    - 输出累加器 o 需要乘以 exp(old_m - new_m) 来修正
    """
    bid = tl.program_id(0)
    
    if bid < B:
        q_start = query_ptr + bid * seq_len_q * head_dim
        k_start = key_ptr + bid * seq_len_kv * head_dim
        v_start = value_ptr + bid * seq_len_kv * head_dim
        mask_start = mask_ptr
        out_start = output_ptr + bid * seq_len_q * head_dim

        for m in range(0, seq_len_q, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q

            # Online-softmax 统计量
            m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
            l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum
            
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
                qk_masked = tl.where(attn_mask, qk_scaled, float('-inf'))

                # ===== Online softmax 更新 =====
                # 计算当前 block 的 max
                m_curr = tl.max(qk_masked, axis=1)  # [BLOCK_M]
                
                # 新的全局 max
                m_new = tl.maximum(m_prev, m_curr)
                
                # 计算修正因子
                alpha = tl.exp(m_prev - m_new)  # [BLOCK_M]
                
                # 计算当前 block 的 softmax 分子 (未归一化)
                p = tl.exp(qk_masked - m_new[:, None])  # [BLOCK_M, BLOCK_N]
                
                # 更新 running sum: l_new = alpha * l_prev + sum(p)
                l_curr = tl.sum(p, axis=1)  # [BLOCK_M]
                l_new = alpha * l_prev + l_curr
                
                # 修正输出累加器: o = alpha * o + p @ V
                acc_o = acc_o * alpha[:, None]
                
                # 加载 V 并累加: p @ V
                for d in range(0, head_dim, BLOCK_D):
                    offs_d = d + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < head_dim
                    
                    v_ptrs = v_start + offs_n[:,None] * head_dim + offs_d[None, :]
                    v = tl.load(v_ptrs, mask=mask_n[:,None] & mask_d[None,:], other=0.0)
                    
                    # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, BLOCK_D]
                    acc_o += tl.dot(p, v)

                # 更新统计量
                m_prev = m_new
                l_prev = l_new

            # 最终归一化: o / l_prev
            acc_o = acc_o / l_prev[:, None]
            
            # 写回输出
            for d in range(0, head_dim, BLOCK_D):
                offs_d = d + tl.arange(0, BLOCK_D)
                mask_d = offs_d < head_dim
                
                out_ptrs = out_start + offs_m[:,None] * head_dim + offs_d[None, :]
                tl.store(out_ptrs, acc_o[:, :], mask=mask_m[:,None] & mask_d[None,:])


def v2(query, key, value, attn_mask):
    """
    SDPA v2 实现 - FlashAttention-style fused attention
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
    grid = (B,)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64
    
    sdpa_kernel_v2[grid](
        query_reshaped, key_reshaped, value_reshaped, attn_mask, output,
        B, seq_len_q, seq_len_kv, head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    output = output.view(batch_size, num_heads_q, seq_len_q, head_dim)
    return output


# ============================================================================
# Triton Kernel v1
# ============================================================================

@triton.jit
def sdpa_kernel_v1(
    query_ptr, key_ptr, value_ptr, mask_ptr, output_ptr,
    attn_ptr,
    B, seq_len_q, seq_len_kv, head_dim,
    scale_factor,
    
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
        # 假设输入是连续的，使用 shape 变量计算偏移
        q_start = query_ptr + bid * seq_len_q * head_dim
        k_start = key_ptr + bid * seq_len_kv * head_dim
        v_start = value_ptr + bid * seq_len_kv * head_dim
        attn_start = attn_ptr + bid * seq_len_q * seq_len_kv
        mask_start = mask_ptr
        out_start = output_ptr + bid * seq_len_q * head_dim

        # target 算 softmax buf
        # ----- | ---- |
        # ----- | ---- | 

        for m in range(0, seq_len_q, BLOCK_M):
            _max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            _exp_sum = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

            offs_m = m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q

            for n in range(0, seq_len_kv, BLOCK_N):
                offs_n = n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len_kv

                # 计算 Q @ K^T 的一个块 [BLOCK_M, BLOCK_N]
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                
                # 遍历 head_dim 维度
                for d in range(0, head_dim, BLOCK_D):
                    offs_d_curr = d + tl.arange(0, BLOCK_D)
                    mask_d = offs_d_curr < head_dim
                    
                    # Q: [BLOCK_M, BLOCK_D]
                    q_ptrs = q_start + offs_m[:,None] * head_dim + offs_d_curr[None, :]
                    q = tl.load(q_ptrs, mask=mask_m[:,None] & mask_d[None,:], other=0.0)
                    
                    # K^T: [BLOCK_D, BLOCK_N] (K 是 [BLOCK_N, BLOCK_D]，转置后)
                    k_ptrs = k_start + offs_n[:,None] * head_dim + offs_d_curr[None, :]
                    k = tl.load(k_ptrs, mask=mask_n[:,None] & mask_d[None,:], other=0.0)
                    
                    # 计算 dot product: Q @ K^T
                    acc += tl.dot(q, tl.trans(k)).to(tl.float32)

                offs_attn = offs_m[:,None]*seq_len_kv + offs_n[None,:]
                attn_fuse_mask = tl.load(mask_start+offs_attn, mask=mask_m[:,None]&mask_n[None,:], other=False)

                # acc 是 softmax 输入的 [M, N] 一小块。考虑外部输入的 mask，一起跑 online-softmax
                acc_scaled = acc * scale_factor
                
                # 对越界位置设为 -inf，使其不影响 softmax 统计量
                acc_masked = tl.where(attn_fuse_mask, acc_scaled, float('-inf'))

                tl.store(attn_start+offs_attn, acc_masked)
                
                block_max = tl.max(acc_masked, axis=-1)
                new_max = tl.maximum(_max, block_max)
                _exp_sum = _exp_sum * tl.exp(_max - new_max) + tl.sum(tl.exp(acc_masked - new_max[:, None]), axis=-1) 
                # triton 是左侧补 1
                # _exp_sum = _exp_sum * tl.exp(_max - new_max) + tl.sum(tl.exp(acc_masked - new_max), axis=-1) 
                _max = new_max

            for n in range(0, seq_len_kv, BLOCK_N):
                # 算 attn score
                offs_n = n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len_kv
                offs_attn = offs_m[:,None]*seq_len_kv + offs_n[None, :]
                # 完整的 mask: [BLOCK_M, BLOCK_N]
                full_mask = mask_m[:,None] & mask_n[None,:]

                attn = tl.load(attn_start+offs_attn, mask=full_mask, other=float('-inf'))

                # 计算 softmax：exp(x - max) / sum_exp
                # 注意：只有有效位置才计算，越界位置保持为0
                attn_score = tl.where(full_mask, tl.exp(attn - _max[:, None]) / _exp_sum[:, None], 0.0)

                tl.store(attn_start+offs_attn, attn_score, mask=full_mask)
        
        # attn_score @ V
        # [seq_len_q, seq_len_kv] @ [seq_len_kv, head_dim]

        for m in range(0, seq_len_q, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q
            
            for n in range(0, head_dim, BLOCK_N):
                offs_n = n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < head_dim

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                
                # 遍历 seq_len_kv 维度
                for k in range(0, seq_len_kv, BLOCK_D):
                    offs_k = k + tl.arange(0, BLOCK_D)
                    mask_k = offs_k < seq_len_kv
                    
                    # attn_score: [BLOCK_M, BLOCK_D]
                    attn_ptrs = attn_start + offs_m[:,None]*seq_len_kv + offs_k[None, :]
                    attn_block = tl.load(attn_ptrs, mask=mask_m[:,None] & mask_k[None,:], other=0.0)
                    
                    # V: [BLOCK_D, BLOCK_N]
                    v_ptrs = v_start + offs_k[:,None] * head_dim + offs_n[None, :]
                    v_block = tl.load(v_ptrs, mask=mask_k[:,None] & mask_n[None,:], other=0.0)
                    
                    acc += tl.dot(attn_block, v_block)
                
                tl.store(out_start+offs_m[:, None]*head_dim + offs_n[None, :], acc, mask=mask_m[:,None]&mask_n[None,:])


def v1(query, key, value, attn_mask):
    """
    SDPA v1 实现
    
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
    
    # 确保所有输入都是 contiguous 的
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_mask = attn_mask.contiguous()
    
    # 默认 scale
    scale = 1.0 / math.sqrt(head_dim)
    
    # 分配输出 (合并 batch 和 num_heads 维度以便处理)
    # 将 (B, H, L, D) reshape 为 (B*H, L, D)
    query_reshaped = query.view(batch_size * num_heads_q, seq_len_q, head_dim)
    key_reshaped = key.view(batch_size * num_heads_kv, seq_len_kv, head_dim)
    value_reshaped = value.view(batch_size * num_heads_kv, seq_len_kv, head_dim)
    
    output = torch.empty_like(query_reshaped)
    
    # 分配 attention score buffer [B*H, seq_len_q, seq_len_kv]
    attn_buffer = torch.zeros(batch_size * num_heads_q, seq_len_q, seq_len_kv, 
                               device=query.device, dtype=torch.float32)
    
    # 启动 kernel: 每个 batch*head 用一个 block
    # 假设输入是连续的，直接使用 shape 变量计算偏移
    B = batch_size * num_heads_q
    grid = (B,)
    
    # 配置 block size
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64
    
    sdpa_kernel_v1[grid](
        query_reshaped, key_reshaped, value_reshaped, attn_mask, output,
        attn_buffer,
        B, seq_len_q, seq_len_kv, head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    if False:
        # 检查 attention score 精度：每行累加和应该接近 1.0
        # attn_buffer 形状: [B*H, seq_len_q, seq_len_kv]
        row_sums = attn_buffer.sum(dim=-1)  # 每行累加和
        print("row_sums" + str(row_sums))
        print(f"[V1 Debug] Attention score row sum - mean: {row_sums.mean():.6f}, "
            f"std: {row_sums.std():.6f}, min: {row_sums.min():.6f}, max: {row_sums.max():.6f}")
        
        # 检查与标准 softmax 的差异（理论上每行和应为 1.0）
        deviation_from_one = (row_sums - 1.0).abs()
        print(f"[V1 Debug] Deviation from 1.0 - mean: {deviation_from_one.mean():.6f}, "
            f"max: {deviation_from_one.max():.6f}")
        
        # 对比 PyTorch 的 attention score
        with torch.no_grad():
            L, S = seq_len_q, seq_len_kv
            scale_factor = 1.0 / math.sqrt(head_dim)
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            
            # 计算参考 attention score (第一个 batch, 第一个 head)
            q_ref = query_reshaped[0:1]  # [1, seq_len_q, head_dim]
            k_ref = key_reshaped[0:1]    # [1, seq_len_kv, head_dim]
            attn_weight_ref = q_ref @ k_ref.transpose(-2, -1) * scale_factor
            attn_weight_ref += attn_bias
            attn_weight_ref = torch.softmax(attn_weight_ref, dim=-1)
            
            # 对比第一个 batch-head 的 attention score
            attn_buffer_first = attn_buffer[0]  # [seq_len_q, seq_len_kv]
            max_diff_attn = (attn_buffer_first - attn_weight_ref[0]).abs().max()
            print(f"[V1 Debug] Attention score max diff vs torch (first head): {max_diff_attn:.6f}")
        
    # reshape 回 4D
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
        for v_name, v_fn in [
            # ("torch_compile", torch_native_sdpa),
            ("v1", v1),
            ("v2", v2),
            ("v3", v3),
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
        line_vals=['tc', 'v1', 'v3'],
        line_names=["TorchCompile", "Triton-v1", "Triton-v3"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-'), ('purple', '-'), ('cyan', '-')],
        ylabel="TFLOPS",  # 注意：SDPA 用 TFLOPS 更合适
        plot_name="sdpa-performance",
        args={'batch_size': 2, 'num_heads': 8, 'head_dim': 64},
    ))
def benchmark(batch_size, num_heads, seq_len, head_dim, provider):
    """SDPA 性能基准测试"""
    # 生成输入
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    
    # 计算 FLOPs (近似)
    # Q @ K^T: 2 * batch * heads * seq_len^2 * head_dim
    # Softmax: 5 * batch * heads * seq_len^2 (approx)
    # Attn @ V: 2 * batch * heads * seq_len^2 * head_dim
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    # 生成 causal mask
    attn_mask = torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool).tril(diagonal=0)
    ms = 0
    try:
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_native_sdpa(query, key, value, attn_mask))
        elif provider == 'tc':
            ms = triton.testing.do_bench(lambda: torch_compile_sdpa(query, key, value, attn_mask))
        elif provider == 'v1':
            ms = triton.testing.do_bench(lambda: v1(query, key, value, attn_mask))
        elif provider == 'v2':
            ms = triton.testing.do_bench(lambda: v2(query, key, value, attn_mask))
        elif provider == 'v3':
            ms = triton.testing.do_bench(lambda: v3(query, key, value, attn_mask))
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
