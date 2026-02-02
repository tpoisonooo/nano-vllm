import torch
import triton
import triton.language as tl
from torch import nn
import torch._dynamo
import os
import math

torch._dynamo.config.recompile_limit = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# ============================================================================
# Ground Truth: SDPA
# ============================================================================

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


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
