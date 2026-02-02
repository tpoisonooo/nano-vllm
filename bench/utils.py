"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.

The following line
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/utils.py#L23
is based on code from Unsloth, located at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

Modifications made by Yanning Chen, 2024.
"""

import functools
from triton.runtime import driver
import math
import torch
import triton
import triton.language as tl


def is_hip() -> bool:
    return torch.version.hip is not None


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def get_npu_core_count(default: int = 20) -> int:
    """Return NPU vector core count.
    Fallback to `default` if Triton runtime or NPU device is unavailable.
    """
    try:
        utils = triton.runtime.driver.active.utils
        props = utils.get_device_properties(0)
        return int(props.get("num_vectorcore", default))
    except Exception:
        return default
    
#   n_cols (已知输入)
#       ↓
#   BLOCK_SIZE (由 n_cols 决定，要覆盖一行或能整除 n_cols)
#       ↓
#   num_warps (由 BLOCK_SIZE 决定，必须满足: num_warps * WARP_SIZE >= BLOCK_SIZE 的并行需求)
#       ↓
#   num_stages (由 SMEM 和寄存器压力决定)
#       ↓
#   ROWS_PER_PROG (由 occupancy 和并行度决定)

class LaunchParam:
    def __init__(self):
        self.block_size = 128
        self.num_warps = 1
        self.num_stages = 2
        self.rows_per_prog = 1

    def next_power_of_2(self, n):
        """返回大于等于 n 的最小 2 的幂"""
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def get_block_size_and_num_warps(self, n_cols, warp_size):
        """
        综合考虑:
        1. 让循环次数 <= 4 (减少同步开销)
        2. num_warps 不要太大 (避免寄存器压力)
        3. 尽量让 n_cols % BLOCK_SIZE 小 (减少 mask 开销)
        """
        if n_cols <= 512:
            # 小 n_cols: 一次处理完，高并行度
            BLOCK_SIZE = max(128, self.next_power_of_2(n_cols))
            num_warps = BLOCK_SIZE // warp_size
            # 但 num_warps 不要超过 8，否则寄存器压力太大
            num_warps = min(num_warps, 8)
        elif n_cols <= 2048:
            # 中等 n_cols: 一次处理完
            BLOCK_SIZE = self.next_power_of_2(n_cols)
            num_warps = BLOCK_SIZE // warp_size
            num_warps = min(num_warps, 8)
        else:
            # 大 n_cols: 循环处理，固定 BLOCK_SIZE
            BLOCK_SIZE = 2048
            num_warps = 8  # 256 threads

        self.block_size = BLOCK_SIZE
        self.num_warps = num_warps

    def deduce(self, n_rows: int, n_cols:int, var_count: int=1, dev: str="cuda:0", target_occupancy:float=0.9) -> LaunchParam:
        # 先推断基本的 block_size 和 num_warps
        
        device = torch.device(dev)
        properties = driver.active.utils.get_device_properties(device.index)
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        MAX_SMEM_PER_BLOCK = properties['MAX_SMEM_PER_BLOCK']
        MAX_BLOCKS_PER_SM = 32

        self.get_block_size_and_num_warps(n_cols, WARP_SIZE)

        # 每个 block 的线程数 = num_warps * WARP_SIZE
        # 约束: 每个 block 的寄存器使用量不能超过 NUM_REGS

        # 每个线程使用的寄存器数 = f(kernel复杂度)
        # 假设每个线程使用 R 个寄存器
        # 则: num_warps * WARP_SIZE * R <= NUM_REGS
        R = var_count * self.block_size

        # 精准模型
        # num_stages <= SIZE_SMEM / (BLOCK_SIZE * bytes_per_element * buffer_count)
        # 经验值
        # num_stages = 4 if SIZE_SMEM > 200000 else 2
        self.num_stages = max(1, min(4, SIZE_SMEM // (self.block_size * 4 * R)))

        # 让 GPU 满载，同时减少 launch overhead

        # 总工作量: n_rows 行需要处理
        # Program (block) 数量应该 >= NUM_SM * occupancy

        # Occupancy 计算:
        # - 每个 SM 能同时运行的 block 数受限于:
        #   a) 寄存器: NUM_REGS / (registers_per_block)
        #   b) 共享内存: SIZE_SMEM / (smem_per_block)
        #   c) 最大 block 数限制 (通常 16 或 32)

        max_blocks_per_sm = min(
            NUM_REGS // (self.num_warps * WARP_SIZE * R),
            SIZE_SMEM // MAX_SMEM_PER_BLOCK,
            MAX_BLOCKS_PER_SM
        )

        # 需要的总 block 数
        total_blocks_needed = NUM_SM * max_blocks_per_sm * target_occupancy
        self.rows_per_prog = max(1, math.ceil(n_rows / total_blocks_needed))

    def print(self):
        # TODO
        pass