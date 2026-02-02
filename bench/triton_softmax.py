import torch
import os
import triton
import triton.language as tl
from triton.runtime import driver
import pdb

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
def softmax_kernel_v1(input_ptr, exp_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # starting row of the program
    row = tl.program_id(0)
    input_ptr = input_ptr + row * n_cols
    output_ptr = output_ptr + row * n_cols
    exp_ptr = exp_ptr + row * n_cols

    _max = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr+cols, mask=mask, other=float('-inf'))
        _max = tl.maximum(x, _max)
    row_max = tl.max(_max, axis=0)
    
    _exp_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(input_ptr + cols, mask=mask, other=float('-inf'))
        _exp = tl.exp(x - row_max)
        tl.store(exp_ptr + cols, _exp, mask=mask)
        _exp_sum += _exp
    exp_sum = tl.sum(_exp_sum, axis=0)

    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(exp_ptr + cols, mask=mask, other=float('-inf'))
        result = x / exp_sum
        tl.store(output_ptr + cols, result, mask=mask)

@triton.jit
def softmax_kernel_v2(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # starting row of the program
    row = tl.program_id(0)
    input_ptr = input_ptr + row * n_cols
    output_ptr = output_ptr + row * n_cols

    # Online softmax: maintain running max and sum as scalars
    # Use 0-d tensor (scalar) by indexing with [None]
    cur_max = tl.full((), float("-inf"), dtype=tl.float32)
    exp_sum = tl.full((), 0.0, dtype=tl.float32)
    
    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr+cols, mask=mask, other=float('-inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(block_max, cur_max)

        exp_sum = exp_sum * tl.exp(cur_max - new_max) + tl.sum(tl.exp(x-new_max), axis=0)
        cur_max = new_max

    for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + cols, mask=mask, other=float('-inf'))
        result = tl.exp(x - cur_max) / exp_sum
        tl.store(output_ptr + cols, result, mask=mask)

@triton.jit
def softmax_kernel_persistant(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    # starting row of the program
    pid = tl.program_id(0)
    row = pid * ROWS_PER_PROG

    if row < n_rows:
        input_ptr = input_ptr + row * n_cols
        output_ptr = output_ptr + row * n_cols

        # Online softmax: maintain running max and sum as scalars
        # Use 0-d tensor (scalar) by indexing with [None]
        cur_max = tl.full((), float("-inf"), dtype=tl.float32)
        exp_sum = tl.full((), 0.0, dtype=tl.float32)
        
        for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            x = tl.load(input_ptr+cols, mask=mask, other=float('-inf'))
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(block_max, cur_max)

            exp_sum = exp_sum * tl.exp(cur_max - new_max) + tl.sum(tl.exp(x-new_max), axis=0)
            cur_max = new_max

        for offset in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            x = tl.load(input_ptr + cols, mask=mask, other=float('-inf'))
            result = tl.exp(x - cur_max) / exp_sum
            tl.store(output_ptr + cols, result, mask=mask)

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

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

def v2(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = 256

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # Create a number of persistent programs.
    # input_ptr, exp_buf, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
    softmax_kernel_v2[(n_rows, 1, 1)](x, y, n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y

def persistant(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = 256

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)


    # 计算超参
    num_warps = 1
    block_size = 128
    if n_cols >= 2048:
        block_size = 2048
        num_warps = 4
    elif n_cols >= 1024:
        block_size = 1024
        num_warps = 2
    elif n_cols >= 512:
        block_size = 256
        num_warps = 1

    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Create a number of persistent programs.
    # input_ptr, exp_buf, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
    softmax_kernel_persistant[(n_rows, 1, 1)](x, y, n_rows, n_cols, BLOCK_SIZE, num_stages, ROWS_PER_PROG)
    return y

def precision_check_all():
    torch.manual_seed(0)
    x_shape = (1823, 781)
    x = torch.rand(x_shape, device=DEVICE)

    output_torch = torch.softmax(x, axis=-1)
    print("PyTorch output sample:", output_torch[0, :5])

    for v_name, v_fn in [
        ("v1", v1),
        ("v2", v2),
    ]:
        y = x.clone()
        output_triton = v_fn(y)
        max_diff = torch.max(torch.abs(output_torch - output_triton))
        print(f"{v_name} max diff vs torch: {max_diff:.6f}")

precision_check_all()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 10240, 128)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['tc', 'torch', 'v1', 'v2'],  # possible values for `line_arg``
        line_names=["tc", "torch", "v1", "v2"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 1024},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'v1':
        ms = triton.testing.do_bench(lambda: v1(x))
    if provider == 'v2':
        ms = triton.testing.do_bench(lambda: v2(x))
    if provider == 'tc':
        ms = triton.testing.do_bench(lambda: torch_compile_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(save_path=os.path.dirname(__file__), print_data=True)