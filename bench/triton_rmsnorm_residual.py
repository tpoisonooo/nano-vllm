import torch
import triton
import triton.language as tl
from torch import nn
import torch._dynamo
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

torch._dynamo.config.recompile_limit = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Ground Truth: torch.compile version (from nanovllm/layers/norm.py)
# ============================================================================


class RMSNormResidualTorchCompile(nn.Module):
    """Ground truth RMSNorm with residual using torch.compile (from nanovllm/layers/norm.py)"""

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
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype).clone()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual


# ============================================================================
# Triton RMSNorm Residual Kernel
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={"BLOCK_SIZE": 128, "ROWS_PER_PROG": 1}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 256, "ROWS_PER_PROG": 1}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 512, "ROWS_PER_PROG": 2}, num_warps=4, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 1024, "ROWS_PER_PROG": 4}, num_warps=8, num_ctas=1
        ),
        triton.Config(
            kwargs={"BLOCK_SIZE": 2048, "ROWS_PER_PROG": 4}, num_warps=8, num_ctas=1
        ),
    ],
    key=["N", "T"],
)
@triton.jit
def _rms_norm_residual_fwd_fused(
    X,  # pointer to the input
    R,  # pointer to the residual
    Y,  # pointer to the output
    R_OUT,  # pointer to the updated residual output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    T,  # max num of rows in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """RMSNorm with residual forward kernel"""
    start_row = tl.program_id(0) * ROWS_PER_PROG

    for row_idx in range(0, ROWS_PER_PROG):
        row = start_row + row_idx
        if row < T:
            offset = row * stride
            X_row = X + offset
            R_row = R + offset
            Y_row = Y + offset
            R_OUT_row = R_OUT + offset

            # First pass: compute x + residual and store to residual output, compute variance
            _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
                r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
                x_plus_r = x + r
                # Store x + residual to residual output (in orig_dtype)
                tl.store(R_OUT_row + cols, x_plus_r, mask=mask)
                _var += x_plus_r * x_plus_r

            var = tl.sum(_var, axis=0) / N
            rstd = 1 / tl.sqrt(var + eps)

            # Second pass: normalize & apply weight
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                w = tl.load(W + cols, mask=mask)
                # Reload x + residual from the stored values
                x_plus_r = tl.load(R_OUT_row + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                x_normalized = x_plus_r * rstd
                tl.store(Y_row + cols, x_normalized * w, mask=mask)


def rms_norm_residual_triton(x, residual, weight, eps=1e-6):
    """RMSNorm with residual using Triton kernel"""
    assert x.is_contiguous()
    assert residual.is_contiguous()
    T, C = x.shape
    out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)

    def grid(meta):
        return (triton.cdiv(T, meta["ROWS_PER_PROG"]),)

    _rms_norm_residual_fwd_fused[grid](
        x, residual, out, residual_out, weight, x.stride(0), C, T, eps
    )

    return out.view(T, C), residual_out.view(T, C)


# ============================================================================
# RMSNorm Residual Module wrapping Triton kernel
# ============================================================================


class RMSNormResidualTriton(nn.Module):
    """RMSNorm with residual using Triton kernel"""

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
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_residual_triton(x, residual, self.weight, self.eps)


# ============================================================================
# Precision Check
# ============================================================================


def precision_check():
    """验证 Triton 实现与 torch.compile 版本的精度一致性"""
    print("=" * 60)
    print("RMSNorm Residual Precision Check")
    print("=" * 60)

    torch.manual_seed(42)

    test_cases = [
        (1024, 512),
        (4096, 1024),
        (10240, 7680),
    ]

    for M, N in test_cases:
        print(f"\nTesting shape: ({M}, {N})")

        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        residual = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

        rms_torch = RMSNormResidualTorchCompile(N, eps=1e-6).to(DEVICE)
        rms_triton = RMSNormResidualTriton(N, eps=1e-6).to(DEVICE)

        rms_triton.weight.data.copy_(rms_torch.weight.data)

        with torch.no_grad():
            output_torch, residual_torch = rms_torch(x.clone(), residual.clone())
            output_triton, residual_triton = rms_triton(x.clone(), residual.clone())

        max_diff_out = torch.max(torch.abs(output_torch - output_triton))
        mean_diff_out = torch.mean(torch.abs(output_torch - output_triton))
        max_diff_res = torch.max(torch.abs(residual_torch - residual_triton))
        mean_diff_res = torch.mean(torch.abs(residual_torch - residual_triton))

        print(f"  Output max difference: {max_diff_out:.2e}")
        print(f"  Output mean difference: {mean_diff_out:.2e}")
        print(f"  Residual max difference: {max_diff_res:.2e}")
        print(f"  Residual mean difference: {mean_diff_res:.2e}")

        if max_diff_out < 1e-4 and max_diff_res < 1e-4:
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
        x_vals=[1 * i for i in range(1, 8)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch_native"],
        line_names=["Triton", "Torch Compile", "Torch Native"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="rms-norm-residual",
        args={"N": 1024, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_rms_norm_residual(
    M, N, dtype, provider, mode="forward", eps=1e-6, device=DEVICE
):
    """Benchmark RMSNorm with residual implementations"""
    x_shape = (M, N)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    residual = torch.randn(x_shape, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "torch_compile":
            rms = RMSNormResidualTorchCompile(N, eps).to(device)
            rms.weight.data = torch.ones(N, dtype=dtype, device=device)
            return rms(x, residual)

        elif provider == "torch_native":
            weight = torch.ones(N, dtype=dtype, device=device)
            x_float = x.float().add_(residual.float())
            residual_out = x_float.to(dtype)
            var = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_float.mul_(torch.rsqrt(var + eps))
            output = x_float.to(dtype) * weight
            return output, residual_out

        elif provider == "triton":
            rms = RMSNormResidualTriton(N, eps).to(device)
            rms.weight.data = torch.ones(N, dtype=dtype, device=device)
            return rms(x, residual)

    # Warmup
    for _ in range(10):
        y_fwd()

    # Benchmark: 3 * M * N for x, residual, and residual_out reads/writes
    # 4 tensors involved: x, residual, output, residual_out
    def gbps(ms):
        return 4 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    precision_check()

    print("\nRunning benchmark...")
    bench_rms_norm_residual.run(save_path=os.path.dirname(__file__), print_data=True)
