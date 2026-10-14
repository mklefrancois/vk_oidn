// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// vk_oidn: lightweight Vulkan denoiser based on Intel(R) Open Image Denoise
//
// DRAFT API (step 1 of the SDK): declarations only, the implementation comes in later steps.
//
// Usage (similar to DLSS / NGX):
//   1. Before creating the VkDevice, enable the extensions and features returned by
//      getRequiredDeviceExtensions() and enableRequiredFeatures().
//   2. Create a Context from the application's Vulkan device (creates the compute pipelines).
//   3. Create a Denoiser for a resolution and set of inputs. This records the upload of the
//      network weights into a command buffer provided by the application.
//   4. Every frame, call Denoiser::cmdDenoise() to record the denoising dispatches into the
//      application's command buffer. The library never submits work or waits on the host.
//
// Synchronization contract for cmdDenoise():
//   - Before: the inputs must have been written and made visible to compute shader reads
//     (e.g. barrier to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_READ_BIT).
//   - After: the output is written by compute shader writes; the application must add a barrier
//     from VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_WRITE_BIT before using it.
//   - Internal resources are synchronized by the library.
//   - The denoiser must not be used by two command buffers executing concurrently.

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace vkoidn {

  enum class Result
  {
    Success,
    ErrorUnsupportedDevice, // a required extension/feature is missing
    ErrorInvalidArgument,
    ErrorOutOfMemory,
    ErrorVulkan,            // a Vulkan call failed
  };

  const char* getResultString(Result result);

  // -----------------------------------------------------------------------------------------------
  // Device requirements
  // -----------------------------------------------------------------------------------------------

  // Appends the device extensions to enable (required ones, plus optional ones supported by the
  // physical device, e.g. VK_KHR_cooperative_matrix)
  void getRequiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                                   PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                                   std::vector<const char*>& extensions);

  // Sets the device features required by the library in the application's feature structures
  // (only sets features to VK_TRUE, never clears them). coopMatFeatures may be null if
  // VK_KHR_cooperative_matrix is not enabled.
  void enableRequiredFeatures(VkPhysicalDeviceFeatures2& features,
                              VkPhysicalDeviceVulkan11Features& features11,
                              VkPhysicalDeviceVulkan12Features& features12,
                              VkPhysicalDeviceVulkan13Features& features13,
                              VkPhysicalDeviceCooperativeMatrixFeaturesKHR* coopMatFeatures);

  // -----------------------------------------------------------------------------------------------
  // Context
  // -----------------------------------------------------------------------------------------------

  struct ContextCreateInfo
  {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;

    // Names of all extensions enabled on the device (determines the optional features used)
    const char* const* enabledExtensions = nullptr;
    uint32_t enabledExtensionCount = 0;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;      // optional
    const VkAllocationCallbacks* allocator = nullptr;    // optional
  };

  class Context
  {
  public:
    // Creates the context and its compute pipelines
    static Result create(const ContextCreateInfo& createInfo, std::unique_ptr<Context>& context);

    virtual ~Context() = default;

    // Whether tensor cores are used (VK_KHR_cooperative_matrix)
    virtual bool isCooperativeMatrixEnabled() const = 0;
  };

  // -----------------------------------------------------------------------------------------------
  // Denoiser
  // -----------------------------------------------------------------------------------------------

  enum class Quality
  {
    Fast,     // smallest network, FP16 accumulation
    Balanced, // base network, FP16 accumulation
    High,     // largest available network, FP32 accumulation
  };

  struct DenoiserCreateInfo
  {
    uint32_t width = 0;
    uint32_t height = 0;

    bool hdr = true;       // color is HDR (otherwise LDR in [0..1])
    bool srgb = false;     // LDR color is sRGB encoded
    bool albedo = false;   // albedo auxiliary input
    bool normal = false;   // normal auxiliary input (requires albedo)
    bool cleanAux = false; // auxiliary inputs are noise-free
    Quality quality = Quality::High;

    // Maximum memory for intermediate data in MB; larger images are processed in tiles.
    // 0 = no limit (single tile).
    uint32_t maxMemoryMB = 0;
  };

  enum class Format
  {
    Float, // 32-bit float channels
    Half,  // 16-bit float channels
  };

  // Image stored in a buffer, specified by its device address
  // (the buffer must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
  struct BufferImage
  {
    VkDeviceAddress address = 0;
    Format format = Format::Float;
    uint32_t channels = 3;        // 1-3 channels are used (e.g. RGBA: channels = 3, pixel stride 16)
    uint32_t pixelByteStride = 0; // 0 = tightly packed
    uint32_t rowByteStride = 0;   // 0 = tightly packed
  };

  // Image specified by a storage image view (VK_IMAGE_LAYOUT_GENERAL, e.g. R32G32B32A32_SFLOAT or
  // R16G16B16A16_SFLOAT). Requires VK_KHR_push_descriptor (core in Vulkan 1.4).
  struct ImageViewImage
  {
    VkImageView view = VK_NULL_HANDLE;
  };

  struct DenoiseInputs
  {
    // Either buffer images or image views must be used for all images
    BufferImage color, albedo, normal, output;
    ImageViewImage colorView, albedoView, normalView, outputView;

    // Input scale for HDR images; NaN = automatic exposure computed on the GPU
    float inputScale = std::numeric_limits<float>::quiet_NaN();
  };

  class Denoiser
  {
  public:
    // Creates the denoiser and records the upload of the network weights into cmd, which must be
    // executed before the first cmdDenoise() call
    static Result create(Context& context, VkCommandBuffer cmd, const DenoiserCreateInfo& createInfo,
                         std::unique_ptr<Denoiser>& denoiser);

    virtual ~Denoiser() = default;

    // Records the denoising dispatches into the command buffer
    virtual Result cmdDenoise(VkCommandBuffer cmd, const DenoiseInputs& inputs) = 0;

    // Device memory used by the denoiser in bytes (weights + intermediate data)
    virtual uint64_t getMemoryUsage() const = 0;
  };

} // namespace vkoidn
