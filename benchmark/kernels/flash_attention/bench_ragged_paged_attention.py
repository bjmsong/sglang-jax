"""
Ref:
https://github.com/sgl-project/sglang-jax/blob/main/benchmark/kernels/flash_attention/bench_flashattention.py
"""

import functools
import time

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)

v6e_peak_flops = 926

def benchmark_backend(
    mode,
    backend_type,
    batch_size,
    seq_len,
    num_heads,
    head_dim=128,
    max_kv_cache_tokens_num=120000,
    page_size=128,
    num_kv_pages_per_block=8,
    num_queries_per_block=32,
):
    if backend_type == "flash":
        if mode == "prefill":
            q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
                create_prefill_uniform_data(
                    batch_size,
                    seq_len,
                    seq_len,
                    max_kv_cache_tokens_num,
                    num_heads,
                    head_dim,
                    page_size=page_size,
                )
            )
        elif mode == "decode":
            q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
                create_decode_uniform_data(
                    batch_size,
                    seq_len,
                    max_kv_cache_tokens_num,
                    num_heads,
                    head_dim,
                    page_size=page_size,
                )
            )

        @functools.partial(
            jax.jit,
            static_argnames=[
                "sm_scale",
                "num_kv_pages_per_block",
                "num_queries_per_block",
            ],
        )
        def jitted_attn(
            q,
            k,
            v,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            sm_scale,
            num_kv_pages_per_block,
            num_queries_per_block,
        ):
            return ragged_paged_attention(
                q,
                k,
                v,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                num_seqs,
                seq_lens,
                sm_scale=sm_scale,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
            )

        attn = functools.partial(
            jitted_attn,
            q,
            k,
            v,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            head_dim**-0.5,
        )
    else:
        raise ValueError(f"Invalid backend type: {backend_type}")

    # Benchmark
    # warm up
    out = attn(num_kv_pages_per_block, num_queries_per_block)
    jax.block_until_ready(out)
    # start benchmark
    times = []
    for i in range(3):
        start = time.perf_counter()
        output = attn(num_kv_pages_per_block, num_queries_per_block)
        jax.block_until_ready(output)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    return avg_time


def main():
    page_size = 128
    bench_modes = ["prefill"]
    num_head_config = [16]   # 2, 4, 8, 16
    seq_len_config = [4096]  # 1024, 2048, 4096
    batch_size_config = [10]  # 1, 2, 4, 8, 10
    head_dim_config = [128]
    # 搜索候选集
    kv_pages_candidates = [4, 8]
    queries_per_block_candidates = [64, 128]
    page_size_candidates = [64, 128]
    all_combined_config = []
    for batch_size in batch_size_config:
        for seq_len in seq_len_config:
            for num_heads in num_head_config:
                for head_dim in head_dim_config:
                    all_combined_config.append(
                        (batch_size, seq_len, num_heads, head_dim)
                    )

    results = []
    for mode in bench_modes:
        print(f"[{mode.upper()}] BENCHMARK RESULTS SUMMARY")
        for i, (batch_size, seq_len, num_heads, head_dim) in enumerate(
            all_combined_config
        ):
            print(f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}")

            # 搜索最优 kernel 配置
            best = {
                "flash_ms": None,
                "fa_tflops": None,
                "utilization": -1.0,
                "num_kv_pages_per_block": None,
                "num_queries_per_block": None,
                "page_size": None,
            }
            for ps in page_size_candidates:
                for kv_pages in kv_pages_candidates:
                    for qpb in queries_per_block_candidates:
                        print(f"page_size={ps}, kv_pages={kv_pages}, qpb={qpb}")
                        flash_time = benchmark_backend(
                            mode,
                            "flash",
                            batch_size,
                            seq_len,
                            num_heads,
                            head_dim=head_dim,
                            page_size=ps,
                            num_kv_pages_per_block=kv_pages,
                            num_queries_per_block=qpb,
                        )

                        total_flops = 2 * 2 * batch_size * num_heads * seq_len * seq_len * head_dim
                        fa_tflops = (total_flops / flash_time) / 1e12
                        utilization = fa_tflops / v6e_peak_flops

                        if utilization > best["utilization"]:
                            best.update(
                                {
                                    "flash_ms": flash_time * 1000,
                                    "fa_tflops": fa_tflops,
                                    "utilization": utilization,
                                    "num_kv_pages_per_block": kv_pages,
                                    "num_queries_per_block": qpb,
                                    "page_size": ps,
                                }
                            )

            results.append(
                {
                    "config": f"B{batch_size}_S{seq_len}_H{num_heads}",
                    "flash_ms": best["flash_ms"],
                    "fa_tflops": best["fa_tflops"],
                    "utilization": best["utilization"],
                    "num_kv_pages_per_block": best["num_kv_pages_per_block"],
                    "num_queries_per_block": best["num_queries_per_block"],
                    "page_size": best["page_size"],
                }
            )
            print()

        print("=" * 80)
        print("-" * 80)

        for r in results:
            print(
                f"{r['config']:<15} {r['flash_ms']:<7.2f} {r['fa_tflops']:<7.2f} {r['utilization']:<5.2f} "
                f"kvpb={r['num_kv_pages_per_block']:<3} qpb={r['num_queries_per_block']:<3} page_size={r['page_size']}"
            )


if __name__ == "__main__":
    main()