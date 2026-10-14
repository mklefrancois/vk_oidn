# vk_oidn

Lightweight Vulkan denoiser based on [Intel® Open Image Denoise](https://github.com/RenderKit/oidn),
designed to be embedded in Vulkan renderers like DLSS: the application enables a few device
features, creates a denoiser, and records `cmdDenoise()` into its own command buffers. The library
never submits work or waits on the host.

- Shaders written in [Slang](https://shader-slang.org), using tensor cores through
  `VK_KHR_cooperative_matrix` (with a fallback for other GPUs).
- Fully fused U-Net layers (convolution + bias + ReLU + pooling / upsampling / concatenation).
- Inputs and outputs as buffers (device addresses) or storage image views.
- Pre-packed network weights embedded in the library (no runtime file loading).

> **Status: work in progress.** This repository is being built step by step:
>
> 1. [x] Repository skeleton, license, draft API (`include/vk_oidn/vk_oidn.h`)
> 2. [ ] Weight packer (`tools/`): OIDN `.tza` weights → GPU-ready blobs
> 3. [x] Slang shaders (`shaders/vk_oidn.slang`), validated in the OIDN fork
> 4. [ ] API implementation (`src/`)
> 5. [ ] Validation against Open Image Denoise and benchmarks (`tests/`)
> 6. [ ] Samples (vk_mini_samples, vk_gltf_renderer)

The reference implementation is the Vulkan backend of the Open Image Denoise fork in
[mklefrancois/oidn-vulkan](https://github.com/mklefrancois/oidn-vulkan) (branch `vulkan`), which
is used to validate the output of this library.

## Building

Requires CMake 3.21+, a C++17 compiler, Python 3 and the Vulkan SDK (headers and `slangc`).

```
cmake -S . -B build
cmake --build build --config Release
```

## Shaders

All kernels are in a single Slang file, [`shaders/vk_oidn.slang`](shaders/vk_oidn.slang), compiled
at build time to SPIR-V (13 kernels, each with and without FP16 float controls) and embedded in
the library. `tools/spirv_float_controls.py` adds the FP16 round-to-nearest-even execution mode,
which Slang has no option for.

The kernels have the same interface as the GLSL kernels of the
[OIDN fork](https://github.com/mklefrancois/oidn-vulkan), which can load them instead of its own
(`OIDN_VULKAN_KERNEL_DIR=<build>/spv`). This was used to validate them: identical results on the
whole accuracy test matrix, all `oidnTest` tests passing, clean validation layer.

Performance on an NVIDIA RTX PRO 4500 Blackwell (`oidnBenchmark`, RT hdr+alb+nrm):

| Resolution | Quality | OIDN CUDA | GLSL kernels | Slang kernels |
|------------|---------|-----------|--------------|---------------|
| 1920x1080  | high    | 8.10 ms   | 8.12 ms      | **7.50 ms**   |
| 3840x2160  | high    | 34.65 ms  | 33.66 ms     | **30.72 ms**  |
| 1920x1080  | fast    | 5.55 ms   | 4.14 ms      | 4.17 ms       |
| 3840x2160  | fast    | 23.41 ms  | 16.50 ms     | 16.50 ms      |

Notes: the kernels are compiled with `-O1` and use `[unroll]` hints rather than `[ForceUnroll]`;
both choices were measurably faster (full front-end unrolling was ~20% slower).

## Weights

The `weights/` folder contains the Open Image Denoise `RT` filter weights converted to a
GPU-ready format by `tools/pack_weights.py` (layer graph + FP16 weight tiles in the layout of the
convolution kernels). They are embedded in the library by default (`VK_OIDN_EMBED_WEIGHTS`,
requires Python 3 at build time). See [weights/README.md](weights/README.md).

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived from Intel Open Image
Denoise; the network weights are from [oidn-weights](https://github.com/RenderKit/oidn-weights).
