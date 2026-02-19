"""
Benchmark script to compare fused Triton kernel vs torch.compile baseline.

Baseline: torch.compile(RMSNorm) + torch.sigmoid + multiply
V1: Fused Triton kernel (1D grid)
"""

import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import numpy as np
import json


# =============================================================================
# Baseline: torch.compile version
# =============================================================================

class RMSNormTorchCompile(torch.nn.Module):
    """Ground truth RMSNorm using torch.compile."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x


def baseline_fused(o, z, norm_weight, eps=1e-6):
    """
    Baseline: RMSNorm + sigmoid gate
    Equivalent to: RMSNorm(o) * sigmoid(z)
    """
    # Create RMSNorm module
    hidden_size = o.shape[-1]
    rms_norm = RMSNormTorchCompile(hidden_size, eps).to(o.device)
    rms_norm.weight.data.copy_(norm_weight)
    
    # RMSNorm
    o_norm = rms_norm(o)
    
    # Sigmoid gate
    gate = torch.sigmoid(z)
    
    # Multiply
    return o_norm * gate


# =============================================================================
# V1: Fused Triton Kernel (1D grid)
# =============================================================================

@triton.jit
def fused_output_kernel_1d(
    o_ptr,
    z_ptr,
    norm_weight_ptr,
    out_ptr,
    hidden_size,
    eps,
    stride_o_seq,
    stride_z_seq,
    stride_out_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """1D grid kernel - one block per sequence position."""
    seq_idx = tl.program_id(0)
    
    o_base = seq_idx * stride_o_seq
    z_base = seq_idx * stride_z_seq
    out_base = seq_idx * stride_out_seq
    
    num_blocks = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_sq = 0.0
    
    for block_idx in range(num_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        
        o_val = tl.load(o_ptr + o_base + offs, mask=mask, other=0.0).to(tl.float32)
        x_sq = o_val * o_val
        sum_sq += tl.sum(x_sq)
    
    mean_sq = sum_sq / hidden_size
    rms = tl.rsqrt(mean_sq + eps)
    
    for block_idx in range(num_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        
        o_val = tl.load(o_ptr + o_base + offs, mask=mask).to(tl.float32)
        norm_w = tl.load(norm_weight_ptr + offs, mask=mask).to(tl.float32)
        o_val = o_val * rms * norm_w
        
        z_val = tl.load(z_ptr + z_base + offs, mask=mask).to(tl.float32)
        gate = tl.sigmoid(z_val)
        o_val = o_val * gate
        
        tl.store(out_ptr + out_base + offs, o_val.to(out_ptr.dtype.element_ty), mask=mask)


def next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def get_launch_config(hidden_size):
    if hidden_size <= 512:
        BLOCK_SIZE = max(128, next_power_of_2(hidden_size))
        num_warps = min(BLOCK_SIZE // 32, 8)
        num_stages = 3
    elif hidden_size <= 2048:
        BLOCK_SIZE = next_power_of_2(hidden_size)
        num_warps = min(BLOCK_SIZE // 32, 8)
        num_stages = 3
    else:
        BLOCK_SIZE = 2048
        num_warps = 8
        num_stages = 2
    return BLOCK_SIZE, num_warps, num_stages


def fused_output_v1(o, z, norm_weight, eps=1e-6):
    """V1: Fused Triton kernel."""
    o = o.contiguous()
    z = z.contiguous()
    seq_len, hidden_size = o.shape
    out = torch.empty_like(o)
    
    BLOCK_SIZE, num_warps, num_stages = get_launch_config(hidden_size)
    grid = (seq_len,)
    
    fused_output_kernel_1d[grid](
        o, z, norm_weight, out,
        hidden_size, eps,
        o.stride(0), z.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_kernel(kernel_fn, o, z, norm_weight, warmup=20, repeats=200):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(o, z, norm_weight)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        _ = kernel_fn(o, z, norm_weight)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / repeats
    return elapsed_ms


def run_benchmark():
    """Run comprehensive benchmark."""
    device = torch.device("cuda:0")
    
    # Test configurations
    hidden_sizes = [512, 1024, 2048, 4096]
    seq_lens = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    results = {hs: {"seq_lens": [], "baseline": [], "v1": [], "speedup": []} 
               for hs in hidden_sizes}
    
    print("=" * 80)
    print("Benchmark: Baseline (torch.compile) vs V1 (Fused Triton)")
    print("=" * 80)
    
    for hidden_size in hidden_sizes:
        print(f"\nHidden Size: {hidden_size}")
        print(f"{'SeqLen':>8} {'Baseline(ms)':>14} {'V1(ms)':>10} {'Speedup':>10} {'Winner':>8}")
        print("-" * 60)
        
        for seq_len in seq_lens:
            # Create test data
            dtype = torch.bfloat16
            o = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
            z = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
            norm_weight = torch.ones(hidden_size, dtype=torch.float32, device=device)
            
            # Benchmark baseline
            try:
                time_baseline = benchmark_kernel(baseline_fused, o, z, norm_weight)
            except Exception as e:
                time_baseline = float('inf')
                print(f"Baseline failed for hidden={hidden_size}, seq_len={seq_len}: {e}")
            
            # Benchmark V1
            try:
                time_v1 = benchmark_kernel(fused_output_v1, o, z, norm_weight)
            except Exception as e:
                time_v1 = float('inf')
                print(f"V1 failed for hidden={hidden_size}, seq_len={seq_len}: {e}")
            
            # Calculate speedup
            if time_v1 > 0 and time_baseline > 0:
                speedup = time_baseline / time_v1
                winner = "V1" if speedup > 1.0 else "Baseline"
            else:
                speedup = 0.0
                winner = "ERR"
            
            print(f"{seq_len:>8} {time_baseline:>14.4f} {time_v1:>10.4f} {speedup:>10.2f}x {winner:>8}")
            
            results[hidden_size]["seq_lens"].append(seq_len)
            results[hidden_size]["baseline"].append(time_baseline)
            results[hidden_size]["v1"].append(time_v1)
            results[hidden_size]["speedup"].append(speedup)
    
    return results


def plot_results(results):
    """Plot benchmark results."""
    hidden_sizes = list(results.keys())
    n_hs = len(hidden_sizes)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, hidden_size in enumerate(hidden_sizes):
        ax = axes[idx]
        data = results[hidden_size]
        seq_lens = data["seq_lens"]
        baseline = np.array(data["baseline"])
        v1 = np.array(data["v1"])
        speedup = np.array(data["speedup"])
        
        # Plot latency
        ax2 = ax.twinx()
        
        line1 = ax.plot(seq_lens, baseline, 'o-', color=colors[0], label='Baseline', linewidth=2, markersize=6)
        line2 = ax.plot(seq_lens, v1, 's-', color=colors[1], label='V1 (Fused)', linewidth=2, markersize=6)
        line3 = ax2.plot(seq_lens, speedup, '^--', color=colors[2], label='Speedup', linewidth=2, markersize=6)
        
        # Add speedup=1 line
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Speedup=1')
        
        ax.set_xlabel('Sequence Length', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax2.set_ylabel('Speedup (x)', fontsize=11, color=colors[2])
        ax2.tick_params(axis='y', labelcolor=colors[2])
        
        ax.set_title(f'Hidden Size = {hidden_size}', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.suptitle('Fused Output Kernel Benchmark: Baseline vs V1', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to benchmark_results.png")
    
    # Create speedup summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, hidden_size in enumerate(hidden_sizes):
        data = results[hidden_size]
        seq_lens = data["seq_lens"]
        speedup = np.array(data["speedup"])
        ax.plot(seq_lens, speedup, 'o-', label=f'Hidden={hidden_size}', 
                linewidth=2, markersize=6, color=colors[idx])
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Speedup=1 (Break-even)')
    ax.fill_between([min(seq_lens), max(seq_lens)], 1.0, 5.0, alpha=0.1, color='green')
    ax.fill_between([min(seq_lens), max(seq_lens)], 0, 1.0, alpha=0.1, color='red')
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Speedup: V1 (Fused Triton) vs Baseline (torch.compile)', fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('speedup_summary.png', dpi=150, bbox_inches='tight')
    print("Speedup summary saved to speedup_summary.png")


def correctness_check():
    """Verify V1 produces correct results compared to baseline."""
    print("\n" + "=" * 80)
    print("Correctness Check")
    print("=" * 80)
    
    device = torch.device("cuda:0")
    hidden_size = 2048
    seq_len = 142
    
    dtype = torch.bfloat16
    torch.manual_seed(42)
    o = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
    z = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
    norm_weight = torch.ones(hidden_size, dtype=torch.float32, device=device)
    eps = 1e-6
    
    # Baseline
    out_baseline = baseline_fused(o, z, norm_weight, eps)
    
    # V1
    out_v1 = fused_output_v1(o, z, norm_weight, eps)
    
    # Compare
    diff = (out_v1.float() - out_baseline.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max diff (V1 vs Baseline): {max_diff:.6f}")
    print(f"Mean diff (V1 vs Baseline): {mean_diff:.6f}")
    
    # Check if results are close (bfloat16 tolerance)
    if max_diff < 0.1:
        print("✓ PASSED: V1 produces correct results")
        return True
    else:
        print("✗ FAILED: Results differ significantly")
        return False


if __name__ == "__main__":
    # First check correctness
    if correctness_check():
        # Then run benchmark
        results = run_benchmark()
        
        # Save results
        with open("benchmark_results_v2.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        plot_results(results)
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        
        for hidden_size, data in results.items():
            speedups = [s for s in data["speedup"] if s > 0]
            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                min_speedup = np.min(speedups)
                print(f"\nHidden Size {hidden_size}:")
                print(f"  Average Speedup: {avg_speedup:.2f}x")
                print(f"  Max Speedup: {max_speedup:.2f}x")
                print(f"  Min Speedup: {min_speedup:.2f}x")
    else:
        print("\nCorrectness check failed, skipping benchmark.")
