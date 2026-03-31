"""
Vector addition operation (m + n).
"""
import torch
import triton
import triton.language as tl

DEVICE=torch.device(f'cuda:{torch.cuda.current_device()}')

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

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)

def benchmark(size: int, provider: str, device: torch.device = DEVICE):
    m = torch.randn(size=(size,), device=device, dtype=torch.float32)
    n = torch.randn(size=(size,), device=device, dtype=torch.float32)
        
    quantiles = [0.5, 0.05, 0.95]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.add(m, n), quantiles=quantiles)

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(m, n), quantiles=quantiles)
        
    # measure GB/s based on ms results. In our kernel, we read from memory 3 times
    gbps = lambda ms: 3 * m.numel() * m.element_size() * 1e-9 / (ms * 1e-3) 
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)
        

def test_add_kernel(size: int, atol: float = 1e-3, rtol: float = 1e-3, device: torch.device = DEVICE):
    torch.manual_seed(42)
    m = torch.randn(size=(size,), device=device)
    n = torch.randn(size=(size,), device=device)
    
    m_plus_n = torch.add(m, n)
    add_kernel_out = add(m, n)
    
    torch.testing.assert_close(actual=add_kernel_out, expected=m_plus_n, rtol=rtol, atol=atol)
    

test_add_kernel(size=96)
test_add_kernel(size=1024)
test_add_kernel(size=4096)
test_add_kernel(size=4097)
benchmark.run(save_path='.', print_data=True)