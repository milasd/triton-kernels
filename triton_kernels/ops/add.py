import torch
import triton
import triton.language as tl

from triton_kernels.config import DEVICE

"""
Vector addition operation (m + n).
"""

# Actual triton dsl
@triton.jit
def add_kernel(
    x_ptr, 
    y_ptr, 
    n_elements, 
    out_ptr, 
    BLOCK_SIZE: tl.constexpr
    ):
    PID = tl.program_id(axis=0)
    
    # get offsets
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # create a mask to prevent that "empty" elements are added when block is not full
    mask = offsets < n_elements
    
    # Load x and y from SDRAM/VRAM/HBM -> SRAM/on-chip memory
    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)
    
    output = x + y
    
    # Store output back to DRAM
    tl.store(out_ptr + offsets, output, mask=mask)
    

# Prepare the data for adding
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # pre-allocate output for inplace update with our kernel.
    output = torch.empty_like(input=x, requires_grad=False)
    
    assert x.device == DEVICE and x.device == y.device
    
    # define our kernel grid size.
    n_elements = torch.numel(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']))
    
    add_kernel[grid](
        x,
        y,
        n_elements,
        output,
        BLOCK_SIZE=1024
    )
    
    return output