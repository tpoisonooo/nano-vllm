import torch
import triton
import triton.language as tl
from torch import nn
import torch._dynamo
import os

torch._dynamo.config.recompile_limit = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# ============================================================================
# Ground Truth: torch.compile version (from nanovllm/layers/norm.py)
# ============================================================================


class RMSNormTorchCompile(nn.Module):
    """Ground truth RMSNorm using torch.compile (from nanovllm/layers/norm.py)"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x


# ============================================================================
# Triton RMSNorm Kernel (inspired by layernorm v4)
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config(
            kwargs={"BLOCK_SIZE": 128, "ROWS_PER_PROG": 1}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 128, "ROWS_PER_PROG": 4}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 128, "ROWS_PER_PROG": 8}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 128, "ROWS_PER_PROG": 64}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 128, "ROWS_PER_PROG": 128}, num_warps=4, num_ctas=1
        ),
    ],
    key=["N", "T"],
)
@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    T,  # max num of rows in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,  # 每个 program 处理的行数
):
    """RMSNorm forward kernel - 仿照 layernorm v4 的 persistent kernel 风格"""
    # 每个 program 负责 ROWS_PER_PROG 行，blockIdx 映射到起始行
    start_row = tl.program_id(0) * ROWS_PER_PROG

    # 处理分配到的每一行
    for row_idx in range(0, ROWS_PER_PROG):
        row = start_row + row_idx
        if row < T:
            # 计算当前行的偏移
            offset = row * stride
            X_row = X + offset
            Y_row = Y + offset

            # Compute RMS: sqrt(mean(x^2))
            _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
                _var += x * x

            var = tl.sum(_var, axis=0) / N
            rstd = 1 / tl.sqrt(var + eps)

            # Normalize & apply weight
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                w = tl.load(W + cols, mask=mask)
                x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
                # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
                x_normalized = x * rstd
                tl.store(Y_row + cols, x_normalized * w, mask=mask)


def rms_norm_triton(x, weight, eps=1e-6):
    """RMSNorm using Triton kernel"""
    assert x.is_contiguous()
    T, C = x.shape
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(T, meta["ROWS_PER_PROG"]),)

    _rms_norm_fwd_fused[grid](x, out, weight, x.stride(0), C, T, eps)
    # , BLOCK_SIZE=128, ROWS_PER_PROG=8, num_warps=4

    return out.view(T, C)


# ============================================================================
# RMSNorm Module wrapping Triton kernel
# ============================================================================


class RMSNormTriton(nn.Module):
    """RMSNorm using Triton kernel"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return rms_norm_triton(x, self.weight, self.eps)


# ============================================================================
# Precision Check
# ============================================================================


def precision_check():
    """验证 Triton 实现与 torch.compile 版本的精度一致性"""
    print("=" * 60)
    print("RMSNorm Precision Check")
    print("=" * 60)

    torch.manual_seed(42)

    # 测试不同大小的输入
    test_cases = [
        (1024, 512),
        (4096, 1024),
        (10240, 7680),
    ]

    for M, N in test_cases:
        print(f"\nTesting shape: ({M}, {N})")

        # Create input
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

        # Create modules
        rms_torch = RMSNormTorchCompile(N, eps=1e-6).to(DEVICE)
        rms_triton = RMSNormTriton(N, eps=1e-6).to(DEVICE)

        # Copy same weights for fair comparison
        rms_triton.weight.data.copy_(rms_torch.weight.data)

        # Forward pass
        with torch.no_grad():
            output_torch = rms_torch(x.clone())
            output_triton = rms_triton(x.clone())

        # Calculate difference
        max_diff = torch.max(torch.abs(output_torch - output_triton))
        mean_diff = torch.mean(torch.abs(output_torch - output_triton))

        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        # Check if results are close
        if max_diff < 1e-4:
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED (max_diff >= 1e-4)")

    print("\n" + "=" * 60)


# ============================================================================
# Benchmark
# ============================================================================


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[512 * i for i in range(64, 1024, 64)],
        line_arg="provider",
        line_vals=["triton", "torch_compile"],
        line_names=["Triton", "Torch Compile"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="rms-norm",
        args={"N": 128, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_rms_norm(M, N, dtype, provider, mode="forward", eps=1e-6, device=DEVICE):
    """Benchmark RMSNorm implementations"""
    # Create data
    x_shape = (M, N)
    x = torch.randn(x_shape, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "torch_compile":
            rms = RMSNormTorchCompile(N, eps).to(device)
            rms.weight.data = torch.ones(N, dtype=dtype, device=device)
            return rms(x)

        elif provider == "torch_native":
            # Native PyTorch implementation without compile
            weight = torch.ones(N, dtype=dtype, device=device)
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
            return (x.float() * torch.rsqrt(var + eps)).to(dtype) * weight

        elif provider == "triton":
            rms = RMSNormTriton(N, eps).to(device)
            rms.weight.data = torch.ones(N, dtype=dtype, device=device)
            return rms(x)

    # Warmup
    for _ in range(10):
        y_fwd()

    # Benchmark
    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # precision_check()

    print("\nRunning benchmark...")
    bench_rms_norm.run(save_path=os.path.dirname(__file__), print_data=True)
