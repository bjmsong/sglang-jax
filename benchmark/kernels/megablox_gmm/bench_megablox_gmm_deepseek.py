import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.gmm.megablox_gmm_backend import gmm

v6e_tflops = 926
PROFILE = True

def gmm_flops(group_sizes: jnp.ndarray, k: int, n: int):
  m = int(group_sizes.sum())
  return 2 * m * k * n

def create_gmm_test_data(
    m: int,
    k: int,
    n: int,
    num_groups: int,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
):
    """Create test data for megablox gmm benchmark."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    lhs = jax.random.normal(keys[0], (m, k), dtype=dtype)
    rhs = jax.random.normal(keys[1], (num_groups, k, n), dtype=dtype)

    return lhs, rhs


def benchmark_backend(
    m: int,
    k: int,
    n: int,
    num_groups: int,
    group_sizes: jnp.ndarray,
    backend_type: str = "megablox",
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Benchmark megablox gmm with given parameters."""

    if backend_type == "megablox":
        lhs, rhs = create_gmm_test_data(m, k, n, num_groups, dtype)

        @functools.partial(
            jax.jit,
            static_argnames=[
                "preferred_element_type",
                "tiling",
            ],
        )
        def jitted_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
        ):
            return gmm(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type=preferred_element_type,
                tiling=tiling,
            )

        gmm_fn = functools.partial(
            jitted_gmm,
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
        )
    else:
        raise ValueError(f"Invalid backend type: {backend_type}")

    try:
        # Benchmark
        # warm up
        out = gmm_fn()
        jax.block_until_ready(out)

        # start benchmark
        times = []
        for i in range(3):
            start = time.perf_counter()
            output = gmm_fn()
            jax.block_until_ready(output)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        return avg_time
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"  Skipping due to memory error: {e}")
            return None
        else:
            raise e


def create_uniform_group_sizes(num_groups: int, group_size: int) -> jnp.ndarray:
    """Create uniform group sizes array."""
    return jnp.array([group_size] * num_groups, dtype=jnp.int32)


def main():
    # m_config = [4096]
    # k_config = [7168]
    # n_config = [4096]
    # num_groups_config = [8]
    # group_size_config = [512]

    # m_config = [1024 * 16]
    # k_config = [5120]
    # n_config = [2048]
    # num_groups_config = [16]
    # group_size_config = [1024]

    # Config of DeepSeek-V3-671B
    batch_size = 16
    seq_len = 4096
    num_experts_per_tok = 8
    num_experts = 256
    hidden_size = 7168
    intermediate_dim = 2048

    m_config = [int(batch_size * seq_len * num_experts_per_tok)]
    k_config = [hidden_size]
    n_config = [intermediate_dim]
    num_groups_config = [num_experts]
    # group_size_config = [int(batch_size * seq_len * num_experts_per_tok / num_experts)]
    
    print("MEGABLOX GMM BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    results = []
    valid_config_count = 0

    # Base ranges for tiling auto-tuning (BM, BK, BN)
    base_tm = [256, 512, 1024, 2048]
    base_tk = [768, 896, 1024, 1152, 1280, 1408, 1536]
    base_tn = [384, 512, 640]

    # best config
    base_tm = [2048]
    base_tk = [1024]
    base_tn = [512]

    for m in m_config:
        for k in k_config:
            for n in n_config:
                for num_groups in num_groups_config:
                    valid_config_count += 1
                    group_sizes = create_uniform_group_sizes(
                        num_groups, m // num_groups
                    )

                    print(
                        f"Config {valid_config_count}: m={m}, k={k}, n={n}, groups={num_groups}, group_size={m//num_groups}"
                    )

                    try:
                        # Build tiling candidates for this config using the provided bases
                        # Filter by simple constraints to avoid obviously invalid tiles
                        tiling_candidates: list[tuple[int, int, int]] = [
                            (tm, tk, tn)
                            for tm in base_tm
                            for tk in base_tk
                            for tn in base_tn
                            if tk <= k and tn <= n and tm <= m
                        ]

                        best = None  # (tflops, util, time_s, tiling)
                        for tiling in tiling_candidates:
                            megablox_time = benchmark_backend(
                                m,
                                k,
                                n,
                                num_groups,
                                group_sizes,
                                backend_type="megablox",
                                tiling=tiling,
                            )

                            if megablox_time is None:
                                # Skip this tiling configuration due to memory error
                                continue

                            tflops = gmm_flops(group_sizes, k, n) / megablox_time * 1e-12
                            tflops_utilization = tflops / v6e_tflops * 100
                            print(f"  Tiling {tiling}: {tflops:.2f} TFLOPS ({tflops_utilization:.2f}% util), {megablox_time * 1000:.2f} ms")

                            if best is None or tflops > best[0]:
                                best = (tflops, tflops_utilization, megablox_time, tiling)

                        if best is not None:
                            print(f"=> Best tiling: {best[3]} | {best[0]:.2f} TFLOPS ({best[1]:.2f}% util), {best[2] * 1000:.2f} ms")
                            results.append(
                                {
                                    "config": f"M{m}_K{k}_N{n}_G{num_groups}",
                                    "TFLOPS": best[0],
                                    "TFLOPS utilization": best[1],
                                    "m": m,
                                    "k": k,
                                    "n": n,
                                    "num_groups": num_groups,
                                    "tiling": best[3],
                                    "time_ms": best[2] * 1000.0,
                                }
                            )

                    except Exception as e:
                        print(f"  ERROR: {e}")

                    print()

    print("=" * 80)
    print("SUMMARY OF ALL RESULTS")
    print("-" * 80)
    print(f"{'Config':<25} {'TFLOPS':<12} {'Util(%)':<10} {'Time(ms)':<10} {'Tiling':<18} {'M':<6} {'K':<6} {'N':<6} {'Groups':<8}")
    print("-" * 80)

    for r in results:
        tiling_str = str(r.get('tiling', 'N/A'))
        time_ms = r.get('time_ms', float('nan'))
        print(
            f"{r['config']:<25} {r['TFLOPS']:<12.2f} {r['TFLOPS utilization']:<10.2f} {time_ms:<10.2f} {tiling_str:<18} {r['m']:<6} {r['k']:<6} {r['n']:<6} {r['num_groups']:<8}"
        )

    # Find best and worst performing configs
    if results:
        best_config = max(results, key=lambda x: x["TFLOPS"]) 
        worst_config = min(results, key=lambda x: x["TFLOPS"]) 

        print("-" * 80)
        print(
            f"Best performance:  {best_config['config']} - {best_config['TFLOPS']:.2f} TFLOPS ({best_config.get('time_ms', float('nan')):.2f} ms), tiling={best_config.get('tiling', 'N/A')}"
        )
        print(
            f"Worst performance: {worst_config['config']} - {worst_config['TFLOPS']:.2f} TFLOPS ({worst_config.get('time_ms', float('nan')):.2f} ms), tiling={worst_config.get('tiling', 'N/A')}"
        )
        print(
            f"Speedup ratio (TFLOPS): {best_config['TFLOPS'] / max(worst_config['TFLOPS'], 1e-9):.2f}x"
        )

        if PROFILE:
            options = jax.profiler.ProfileOptions()
            options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE"}
            with jax.profiler.trace("/tmp/profile-data", profiler_options=options):  # create_perfetto_link=True
                parts = best_config['config'].split("_")
                m = int(parts[0][1:])
                k = int(parts[1][1:])
                n = int(parts[2][1:])
                num_groups = best_config['num_groups']
                group_sizes = create_uniform_group_sizes(num_groups, m // num_groups)
                tiling = best_config['tiling']
                megablox_time = benchmark_backend(
                    m,
                    k,
                    n,
                    num_groups,
                    group_sizes,
                    backend_type="megablox",
                    tiling=tiling,
                )

                tflops = gmm_flops(group_sizes, k, n) / megablox_time * 1e-12
                tflops_utilization = tflops / v6e_tflops * 100
                print(f"  Tiling {tiling}: {tflops:.2f} TFLOPS ({tflops_utilization:.2f}% util), {megablox_time * 1000:.2f} ms")

if __name__ == "__main__":
    main()
