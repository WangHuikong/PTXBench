import threading
import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


_THREADS_PER_BLOCK = 256


class ScaleShiftKernel:
    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        scale: cutlass.Float32,
        shift: cutlass.Int32,
        y: cute.Tensor,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        idx = block_idx * _THREADS_PER_BLOCK + thread_idx

        if idx < x.shape[0]:
            y[idx] = x[idx] * scale + shift.to(cutlass.Float32)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        scale: cutlass.Float32,
        shift: cutlass.Int32,
        y: cute.Tensor,
        stream: cuda.CUstream,
    ):
        blocks = (x.shape[0] + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        self.kernel(x, scale, shift, y).launch(
            grid=(blocks, 1, 1),
            block=(_THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )


_kernel = ScaleShiftKernel()
_compiled_kernel = None
_compile_lock = threading.Lock()

# This is necessary for kernel evaluation
def _get_compiled_kernel(x_cute, scale, shift, y_cute, stream):
    """Compile once, then reuse the returned executable on later calls."""
    global _compiled_kernel

    if _compiled_kernel is None:
        with _compile_lock:
            if _compiled_kernel is None:
                _compiled_kernel = cute.compile(
                    _kernel, x_cute, scale, shift, y_cute, stream
                )
    return _compiled_kernel


def run(x_torch, scale, shift, y_torch):
    """Compute ``y = x * scale + shift`` on PyTorch CUDA tensors."""

    # This is necessary to ensure the correct GPU is used, especially in multi-GPU setups.
    torch.cuda.set_device(x_torch.device)

    x_cute = from_dlpack(x_torch, assumed_align=16)
    y_cute = from_dlpack(y_torch, assumed_align=16)

    # Preserve stride 1 while making the vector length a runtime dimension, so
    # the same compiled callable can be reused for differently sized vectors.
    x_cute.mark_layout_dynamic(leading_dim=0)
    y_cute.mark_layout_dynamic(leading_dim=0)

    scale = cutlass.Float32(scale)
    shift = cutlass.Int32(shift)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = _get_compiled_kernel(x_cute, scale, shift, y_cute, stream)
    compiled_kernel(x_cute, scale, shift, y_cute, stream)
