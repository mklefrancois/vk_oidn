// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vk_oidn/vk_oidn.h"
#include <cstring>

namespace vkoidn {

  const char* getResultString(Result result)
  {
    switch (result)
    {
    case Result::Success:                return "success";
    case Result::ErrorUnsupportedDevice: return "unsupported device";
    case Result::ErrorInvalidArgument:   return "invalid argument";
    case Result::ErrorOutOfMemory:       return "out of memory";
    case Result::ErrorVulkan:            return "Vulkan error";
    default:                             return "unknown error";
    }
  }

  void getRequiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                                   PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                                   std::vector<const char*>& extensions)
  {
    auto enumerateExtensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
      getInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties"));
    if (!enumerateExtensions)
      return;

    uint32_t count = 0;
    enumerateExtensions(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    enumerateExtensions(physicalDevice, nullptr, &count, props.data());

    auto addIfSupported = [&](const char* name)
    {
      for (const auto& p : props)
      {
        if (strcmp(p.extensionName, name) == 0)
        {
          for (const char* e : extensions)
            if (strcmp(e, name) == 0)
              return; // already enabled
          extensions.push_back(name);
          return;
        }
      }
    };

    // Optional: tensor cores and image view inputs/outputs
    addIfSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    addIfSupported(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  }

  void enableRequiredFeatures(VkPhysicalDeviceFeatures2& features,
                              VkPhysicalDeviceVulkan11Features& features11,
                              VkPhysicalDeviceVulkan12Features& features12,
                              VkPhysicalDeviceVulkan13Features& features13,
                              VkPhysicalDeviceCooperativeMatrixFeaturesKHR* coopMatFeatures)
  {
    features.features.shaderInt64 = VK_TRUE;
    features11.storageBuffer16BitAccess = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.shaderFloat16 = VK_TRUE;
    features12.vulkanMemoryModel = VK_TRUE;
    features12.vulkanMemoryModelDeviceScope = VK_TRUE;
    features13.subgroupSizeControl = VK_TRUE;
    features13.computeFullSubgroups = VK_TRUE;
    if (coopMatFeatures)
      coopMatFeatures->cooperativeMatrix = VK_TRUE;
  }

  // TODO (step 4): implementation of Context and Denoiser
  Result Context::create(const ContextCreateInfo& createInfo, std::unique_ptr<Context>& context)
  {
    context.reset();
    return Result::ErrorUnsupportedDevice;
  }

  Result Denoiser::create(Context& context, VkCommandBuffer cmd, const DenoiserCreateInfo& createInfo,
                          std::unique_ptr<Denoiser>& denoiser)
  {
    denoiser.reset();
    return Result::ErrorUnsupportedDevice;
  }

} // namespace vkoidn
