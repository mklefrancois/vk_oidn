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

**Documentation:**
[architecture](docs/architecture.md) (how it works, with diagrams) ·
[validation and performance](docs/validation_and_performance.md) (all measurements and how they
were made, issues found, experiments).

**Status:** version 0.x: the API may still change between minor versions. Releases with the
prebuilt SDK packages are on the [releases page](https://github.com/nvpro-samples/vk_oidn/releases).
Example: the `oidn_denoiser` sample of vk_mini_samples (path-traced Cornell box, denoiser with
albedo and normal guides).

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

// Alternatively, storage image views (VK_IMAGE_LAYOUT_GENERAL, RGBA32F/RGBA16F, alpha preserved):
inputs = {};
inputs.colorView  = {colorView};
inputs.albedoView = {albedoView};
inputs.normalView = {normalView};
inputs.outputView = {outputView, VK_FORMAT_R16G16B16A16_SFLOAT}; // format: exact FP16 rounding
denoiser->cmdDenoise(cmd, inputs);
```

See [`tests/denoise_test.cpp`](tests/denoise_test.cpp) for a complete example. The validation
can be reproduced with [`tests/scripts`](tests/scripts) (image generator and validation script, see
the [validation document](docs/validation_and_performance.md)). It is also a
command-line tool: `vk_oidn_denoise_test --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -o out.pfm`
(`-r ref.pfm` compares with a reference, `-n 50` measures the GPU time, `--images` uses storage
image views instead of buffers).

Validation: on 22 configurations (HDR/LDR/sRGB, albedo/normal/clean aux, all qualities, half
images, fixed input scale, tiled), the output matches the OIDN CPU reference (as accurately as
OIDN's CUDA backend), and is bit-identical to the Vulkan backend of the OIDN fork when the
sub-pixel optimization is disabled. The validation layer reports no errors.

GPU time on an NVIDIA RTX PRO 4500 Blackwell (hdr + albedo + normal):

| Resolution | Quality | vk_oidn   | OIDN 2.5.1 CUDA |
|------------|---------|-----------|-----------------|
| 1920x1080  | high    | 5.01 ms   | 8.10 ms         |
| 1920x1080  | fast    | 2.89 ms   | 5.55 ms         |
| 3840x2160  | high    | 22.77 ms  | 34.65 ms        |
| 3840x2160  | fast    | 12.55 ms  | 23.41 ms        |

## Adding vk_oidn to a project

**Prebuilt SDK package** (recommended): each [release](https://github.com/nvpro-samples/vk_oidn/releases)
has a package per platform with the header, the static libraries (kernels and weights embedded)
and the CMake package files. Nothing of vk_oidn is compiled; only the Vulkan headers are needed.

| Package                                 | Contents                                                   |
|-----------------------------------------|------------------------------------------------------------|
| `vk_oidn-<version>-windows-x64.zip`     | `vk_oidn.lib` (Release) and `vk_oidnd.lib` (Debug), MSVC `/MD` runtime, Visual Studio 2022 or newer |
| `vk_oidn-<version>-linux-x64.tar.gz`    | `libvk_oidn.a` (Release, position independent), GCC 13 (Ubuntu 24.04) |

```cmake
include(FetchContent)
if(WIN32)
  set(VK_OIDN_PACKAGE vk_oidn-0.1.3-windows-x64.zip)
else()
  set(VK_OIDN_PACKAGE vk_oidn-0.1.3-linux-x64.tar.gz)
endif()
FetchContent_Declare(vk_oidn_sdk
  URL https://github.com/nvpro-samples/vk_oidn/releases/download/v0.1.3/${VK_OIDN_PACKAGE})
FetchContent_MakeAvailable(vk_oidn_sdk)
find_package(vk_oidn 0.1 CONFIG REQUIRED PATHS ${vk_oidn_sdk_SOURCE_DIR} NO_DEFAULT_PATH)
target_link_libraries(my_app PRIVATE vk_oidn::vk_oidn)
```

Applications built as RelWithDebInfo or MinSizeRel use the Release library. The checksums of the
packages are in `SHA256SUMS.txt` of each release (use them with `URL_HASH SHA256=...`).

**From the sources**: `FetchContent_Declare(vk_oidn GIT_REPOSITORY https://github.com/nvpro-samples/vk_oidn.git GIT_TAG v0.1.3)`
and `FetchContent_MakeAvailable(vk_oidn)` give the same `vk_oidn::vk_oidn` target, built with the
application (requires Python 3 and `slangc`, see below).

## Building

[![Build](https://github.com/nvpro-samples/vk_oidn/actions/workflows/build.yml/badge.svg)](https://github.com/nvpro-samples/vk_oidn/actions/workflows/build.yml)

Windows and Linux. Requires CMake 3.21+, a C++17 compiler, Python 3 and the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (headers and `slangc`; tested with 1.4.363.0,
Slang 2026.17). On Linux, the Vulkan loader library is also needed (`libvulkan-dev`).

```
cmake -S . -B build
cmake --build build --config Release
cmake --install build --config Release --prefix <sdk folder>   # creates the SDK package
```

The `slangc` of some Linux distribution packages is too old for the kernels: CMake then stops
with a message. Set `VULKAN_SDK` to the LunarG SDK (`source <sdk>/setup-env.sh`), or
`VK_OIDN_SLANGC` to a recent `slangc` from the [Slang releases](https://github.com/shader-slang/slang/releases).

Every push is built on Windows and Linux by [GitHub Actions](.github/workflows/build.yml), with
warnings as errors. On Linux, the denoiser is also run on lavapipe (Mesa's CPU Vulkan driver) in
11 configurations, with buffers and with image views, and compared to the official Open Image
Denoise CPU device, with the Khronos validation layer enabled (see
[docs/validation_and_performance.md](docs/validation_and_performance.md)). Each build also creates
the SDK packages and links a small application against them
([tests/package](tests/package)); for a version tag `v*`, the packages are attached to the
release.

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

## Debugging environment variables

| Variable             | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `VK_OIDN_PROFILE=1`  | print per-layer GPU times when the denoiser is destroyed             |
| `VK_OIDN_COOPMAT=0/1`| disable / force cooperative matrices (default: NVIDIA only)          |
| `VK_OIDN_CM2=1`      | experimental `VK_NV_cooperative_matrix2` convolutions (slower, see the docs) |
| `VK_OIDN_SUBPIXEL=0` | disable the sub-pixel decomposition of the upsampling layers          |
| `VK_OIDN_CONV_VARIANT=mx2_nf1` | force 1 (or 2) output blocks per workgroup in the convolutions (tuning) |
| `VK_OIDN_KERNEL_DIR=<dir>` | load `<dir>/<kernel>.spv` instead of the embedded kernel when it exists (kernel development) |

## Weights

The `weights/` folder contains the Open Image Denoise `RT` filter weights converted to a
GPU-ready format by `tools/pack_weights.py` (layer graph + FP16 weight tiles in the layout of the
convolution kernels). They are embedded in the library by default (`VK_OIDN_EMBED_WEIGHTS`,
requires Python 3 at build time). See [weights/README.md](weights/README.md).

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived from Intel Open Image
Denoise; the network weights are from [oidn-weights](https://github.com/RenderKit/oidn-weights).
