# Tensor core micro-benchmarks

Small standalone measurements behind the "Tensor core peak, and an experiment with cooperative
vectors" section of [docs/validation_and_performance.md](../../docs/validation_and_performance.md).
They are not part of the vk_oidn build.

| File | Measures |
|------|----------|
| `bench.cpp` | Harness: runs a SPIR-V compute kernel with buffers given by device address and measures its GPU time (`bench --info` lists the cooperative matrix and vector capabilities) |
| `peak.comp` | Peak `VK_KHR_cooperative_matrix` throughput (FP16, FP8 E4M3, INT8): register-only multiply-add chains |
| `coopvec_peak.comp` | `VK_NV_cooperative_vector` matrix-vector chains with the input in registers |
| `conv_coopvec.comp` | A 3x3 convolution with cooperative vectors, and variants isolating the products and the memory traffic |
| `run_all.sh` | Builds the kernels with glslang and runs all measurements |

```
cmake -S tests/microbench -B build_microbench
cmake --build build_microbench --config Release
bash tests/microbench/run_all.sh build_microbench/Release/bench    # Linux: build_microbench/bench
```

Requires an NVIDIA GPU with `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_vector` and
`VK_EXT_shader_float8`, the Vulkan SDK (glslang) and Python 3.
