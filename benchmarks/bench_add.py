"""Benchmark for vector addition kernel."""
import torch
import triton
import triton.testing
from triton_kernels.config import DEVICE
from triton_kernels.ops.add import add


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


if __name__ == "__main__":
    benchmark.run(save_path='./benchmarks/results', print_data=True)
