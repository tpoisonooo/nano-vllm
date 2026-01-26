import argparse
import itertools
import torch
import triton
import triton.language as tl
import triton.profiler as proton
from contextlib import contextmanager
from torch.nn import functional as F

from typing import Optional
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# output = (x - mean) / (var(x) + epsilon) * w + b

@triton.jit
def _ln_stats(X, Mean, Rstd, stride, N, EPS,
              BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    X   = X   + row * stride
    # 第一趟：均值
    cols = tl.arange(0, BLOCK_SIZE)
    msum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        msum += a                # 把当前 tile 的和累进来
    mean = tl.sum(msum, axis=0) / N              # 再归约成标量
    tl.store(Mean + row, mean)

    # 第二趟：方差 + rstd
    vsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        vsum += x * x
    var = tl.sum(vsum, axis=0) / N
    rstd = 1 / tl.sqrt(var + EPS)
    tl.store(Rstd + row, rstd)

@triton.jit
def _ln_apply(X, W, B, Out, stride, N, Mean, Rstd,
              BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    X   = X   + row * stride
    Out = Out + row * stride
    mean  = tl.load(Mean + row)
    rstd  = tl.load(Rstd + row)

    cols = tl.arange(0, BLOCK_SIZE)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Out + cols, y, mask=mask)

def layer_norm_triton_v1(x, weight, bias, eps=1e-6):
    assert x.is_contiguous()
    T, C = x.shape
    mean = torch.empty((T,), device=x.device, dtype=x.dtype)
    rstd = torch.empty((T,), device=x.device, dtype=x.dtype)
    out  = torch.empty_like(x)

    BLOCK_SIZE = 128           # 128 或 256 都行，只要 ≤C 且 2 的幂
    grid = (T,)
    _ln_stats[grid](x, mean, rstd, C, C, eps, BLOCK_SIZE)
    _ln_apply[grid](x, weight, bias, out, C, C, mean, rstd,
                    BLOCK_SIZE)
    return out.view(T, C)


@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=8, num_ctas=1),

        # triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_ctas=2),
    ],
    key=['N'],
)
@triton.jit
def _layer_norm_fwd_fused_v2(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    tl.store(Mean + row, mean)

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

def layer_norm_triton_v2(x, weight, bias, eps=1e-6):
    assert x.is_contiguous()
    T, C = x.shape
    mean = torch.empty((T,), device=x.device, dtype=x.dtype)
    rstd = torch.empty((T,), device=x.device, dtype=x.dtype)
    out  = torch.empty_like(x)

    grid = (T,)
    _layer_norm_fwd_fused_v2[grid](x, out, weight, bias, mean, rstd, x.stride(0), C, eps)
    
    return out.view(T, C)



@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=8, num_ctas=1),

        # triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_ctas=2),
    ],
    key=['N'],
)
@triton.jit
def _layer_norm_fwd_fused_v3(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += x
        # 数值稳定性较差
        _var += x * x

    mean = tl.sum(_mean, axis=0) / N
    var = tl.sum(_var, axis=0) / N - mean * mean  # 修正：E[x²] - E[x]²

    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

def layer_norm_triton_v3(x, weight, bias, eps=1e-6):
    assert x.is_contiguous()
    T, C = x.shape
    mean = torch.empty((T,), device=x.device, dtype=x.dtype)
    rstd = torch.empty((T,), device=x.device, dtype=x.dtype)
    out  = torch.empty_like(x)

    grid = (T,)
    _layer_norm_fwd_fused_v2[grid](x, out, weight, bias, mean, rstd, x.stride(0), C, eps)
    
    return out.view(T, C)


@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128, 'ROWS_PER_PROG': 1}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 256, 'ROWS_PER_PROG': 1}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 512, 'ROWS_PER_PROG': 2}, num_warps=4, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 1024, 'ROWS_PER_PROG': 4}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 2048, 'ROWS_PER_PROG': 4}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 4096, 'ROWS_PER_PROG': 8}, num_warps=8, num_ctas=1),
        triton.Config(kwargs={'BLOCK_SIZE': 8192, 'ROWS_PER_PROG': 8}, num_warps=8, num_ctas=1),

        # triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_ctas=2),
    ],
    key=['N', 'T'],
)
@triton.jit
def _layer_norm_fwd_fused_v4(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    T,  # max num of rows in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,  # 每个 program 处理的行数
):
    # 每个 program 负责 ROWS_PER_PROG 行，blockIdx 映射到起始行
    start_row = tl.program_id(0) * ROWS_PER_PROG
    # 处理分配到的每一行
    for row_idx in range(0, ROWS_PER_PROG):
        row = start_row + row_idx
        if row < T:
                
            # 原逻辑，row 改为循环变量
            offset = row * stride
            X_row = X + offset
            Y_row = Y + offset
            
            # Compute mean & var (融合版)
            _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(X_row + cols, mask=mask, other=0.).to(tl.float32)
                _mean += x
                _var += x * x
            
            mean = tl.sum(_mean, axis=0) / N
            var = tl.sum(_var, axis=0) / N - mean * mean
            rstd = 1 / tl.sqrt(var + eps)
            
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
            
            # Normalize & transform
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                w = tl.load(W + cols, mask=mask)
                b = tl.load(B + cols, mask=mask)
                x = tl.load(X_row + cols, mask=mask, other=0.).to(tl.float32)
                x_hat = (x - mean) * rstd
                tl.store(Y_row + cols, x_hat * w + b, mask=mask)

def layer_norm_triton_v4(x, weight, bias, eps=1e-6):
    # persistant kernel
    assert x.is_contiguous()
    T, C = x.shape
    mean = torch.empty((T,), device=x.device, dtype=x.dtype)
    rstd = torch.empty((T,), device=x.device, dtype=x.dtype)
    out  = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(T, meta['ROWS_PER_PROG']),)

    _layer_norm_fwd_fused_v4[grid](x, out, weight, bias, mean, rstd, x.stride(0), C, T, eps)
    
    return out.view(T, C)


def precision_check():
    torch.manual_seed(0)
    x_shape = (10240, 7680)
    w_shape = (x_shape[-1], )
    x = torch.rand(x_shape, device=DEVICE)
    y = x.clone()
    w = torch.rand(w_shape, device=DEVICE)
    b = torch.rand(w_shape, device=DEVICE)

    output_torch = F.layer_norm(x, w_shape, w, b, 1e-6)
    print(output_torch)

    output_triton = layer_norm_triton_v4(y, w, b)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}')

precision_check()
import time
time.sleep(6)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 64)],
        line_arg='provider',
        line_vals=['v2', 'v3', 'v4', 'torch'],
        line_names=['v2', 'v3', 'v4', 'Torch'],
        styles=[('green', '-'), ('yellow', '-'), ('pink', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='layer-norm',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        # if provider == "v1":
        #     return layer_norm_triton_v1(x, weight, bias)  # noqa: F811, E704
        
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "v2":
            return layer_norm_triton_v2(x, weight, bias)  # noqa: F811, E704
        
        if provider == "v3":
            return layer_norm_triton_v3(x, weight, bias)  # noqa: F811, E704

        if provider == "v4":
            return layer_norm_triton_v4(x, weight, bias)  # noqa: F811, E704

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

bench_layer_norm.run(save_path='.', print_data=True)