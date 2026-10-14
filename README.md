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
> 4. [x] API implementation (`src/`) with buffer images; storage image views: in progress
> 5. [ ] Validation against Open Image Denoise and benchmarks (`tests/`)
> 6. [ ] Samples (vk_mini_samples, vk_gltf_renderer)

The reference implementation is the Vulkan backend of the Open Image Denoise fork in
[mklefrancois/oidn-vulkan](https://github.com/mklefrancois/oidn-vulkan) (branch `vulkan`), which
is used to validate the output of this library.

## Usage

```cpp
#include <vk_oidn/vk_oidn.h>

// 1. Before creating the VkDevice: enable what vk_oidn needs
vkoidn::getRequiredDeviceExtensions(instance, physicalDevice, vkGetInstanceProcAddr, extensions);
vkoidn::enableRequiredFeatures(features2, features11, features12, features13,
                               hasCoopMat ? &coopMatFeatures : nullptr);

// 2. Context (once per device)
vkoidn::ContextCreateInfo contextInfo{instance, physicalDevice, device,
                                      vkGetInstanceProcAddr, vkGetDeviceProcAddr,
                                      extensions.data(), uint32_t(extensions.size())};
std::unique_ptr<vkoidn::Context> context;
vkoidn::Context::create(contextInfo, context);

// 3. Denoiser (per resolution/configuration): records the weight upload into cmd
vkoidn::DenoiserCreateInfo info;
info.width = width;  info.height = height;
info.hdr = true;     info.albedo = true;  info.normal = true;
info.quality = vkoidn::Quality::High;
std::unique_ptr<vkoidn::Denoiser> denoiser;
vkoidn::Denoiser::create(*context, cmd, info, denoiser);

// 4. Every frame: records the dispatches into the application's command buffer
vkoidn::DenoiseInputs inputs;
inputs.color  = {colorAddress,  vkoidn::Format::Float, 3, 16}; // e.g. RGBA32F, pixel stride 16
inputs.albedo = {albedoAddress, vkoidn::Format::Float, 3, 16};
inputs.normal = {normalAddress, vkoidn::Format::Float, 3, 16};
inputs.output = {outputAddress, vkoidn::Format::Float, 3, 16};
denoiser->cmdDenoise(cmd, inputs);
// barrier: compute shader write -> your use of the output
```

See [`tests/denoise_test.cpp`](tests/denoise_test.cpp) for a complete example. It is also a
command-line tool: `vk_oidn_denoise_test --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -o out.pfm`
(`-r ref.pfm` compares with a reference, `-n 50` measures the GPU time).

Validation: on 22 configurations (HDR/LDR/sRGB, albedo/normal/clean aux, all qualities, half
images, fixed input scale, tiled), the output is bit-identical to the Vulkan backend of the OIDN
fork and matches the OIDN CPU reference. The validation layer reports no errors.

GPU time on an NVIDIA RTX PRO 4500 Blackwell (hdr + albedo + normal):

| Resolution | Quality | vk_oidn   | OIDN 2.5.1 CUDA |
|------------|---------|-----------|-----------------|
| 1920x1080  | high    | 7.31 ms   | 8.10 ms         |
| 1920x1080  | fast    | 3.95 ms   | 5.55 ms         |
| 3840x2160  | high    | 30.30 ms  | 34.65 ms        |
| 3840x2160  | fast    | 16.09 ms  | 23.41 ms        |

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
