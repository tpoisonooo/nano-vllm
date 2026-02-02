import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.compile
def torch_compile_softmax(x: torch.Tensor):
    # x shape (m,n)
    # read (m,n), write (m)
    x_max = x.max(dim=1)[0]
    # read (m,n), write (m,n)
    neg = x - x_max[:, None]
    # read (m,n), write (m,n)
    x_exp = torch.exp(neg)
    
    # read (m,n), write(m)
    denomator = x_exp.sum(dim=1)
    # read (m,n), write(m,n)
    ret = x_exp / denomator[:, None]
    # read 5mn, write 2m+3mn
    return ret


@triton.jit
def softmax_kernel_v1(input_ptr, exp_buf, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # starting row of the program
    row = tl.program_id(0)
    input_ptr = input_ptr + row * n_cols
    output_ptr = output_ptr + row * n_cols

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        row = tl.load(input_ptr+cols, mask=mask, other=0.0)
        _sum += row
    sum = tl.sum(_sum, axis=0)

    _exp_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        row = tl.load(input_ptr + cols, mask=mask, other=0.0)
        _exp = tl.exp(row - sum)
        tl.store(exp_buf + cols, _exp, mask=mask)
        _exp_sum += _exp
    exp_sum = tl.sum(_exp_sum, axis=0)

    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        row = tl.load(exp_buf + cols, mask=mask, other=0.0)
        result = tl.fdiv(row, exp_sum)
        tl.store(output_ptr + cols, result, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def v1(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = 256

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate exp buf
    buf = torch.empty_like(x)

    # Allocate output
    y = torch.empty_like(x)

    # Create a number of persistent programs.
    # input_ptr, exp_buf, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr

    softmax_kernel_v1[(n_rows, 1, 1)](x, buf, y, n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


def precision_check_all():
    torch.manual_seed(0)
    x_shape = (1823, 781)
    w_shape = (x_shape[-1],)
    x = torch.rand(x_shape, device=DEVICE)
    w = torch.rand(w_shape, device=DEVICE)
    b = torch.rand(w_shape, device=DEVICE)

    output_torch = torch.softmax(x, axis=-1)
    print("PyTorch output sample:", output_torch[0, :5])

    for v_name, v_fn in [
        ("v1", v1),
    ]:
        y = x.clone()
        output_triton = v_fn(y, w, b)
        max_diff = torch.max(torch.abs(output_torch - output_triton))
        print(f"{v_name} max diff vs torch: {max_diff:.6f}")

precision_check_all()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100, 16)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch_compile', 'torch', 'v1'],  # possible values for `line_arg``
        line_names=["Triton", "Torch"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'v1':
        ms = triton.testing.do_bench(lambda: v1(x))
    if provider == 'torch_compile':
        ms = triton.testing.do_bench(lambda: torch_compile_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(save_path=os.path.dirname(__file__), print_data=True)