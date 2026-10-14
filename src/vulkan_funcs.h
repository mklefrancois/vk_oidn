// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

namespace vkoidn {

  // Instance-level functions used by the library
  #define VKOIDN_INSTANCE_FUNCS(F)                       \
    F(vkGetPhysicalDeviceProperties)                     \
    F(vkGetPhysicalDeviceProperties2)                    \
    F(vkGetPhysicalDeviceMemoryProperties)               \
    F(vkGetDeviceProcAddr)

  #define VKOIDN_INSTANCE_FUNCS_OPT(F)                   \
    F(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)

  // Device-level functions used by the library
  #define VKOIDN_DEVICE_FUNCS(F)                         \
    F(vkCreateBuffer)                                    \
    F(vkDestroyBuffer)                                   \
    F(vkGetBufferMemoryRequirements)                     \
    F(vkAllocateMemory)                                  \
    F(vkFreeMemory)                                      \
    F(vkBindBufferMemory)                                \
    F(vkMapMemory)                                       \
    F(vkUnmapMemory)                                     \
    F(vkCreateDescriptorSetLayout)                       \
    F(vkDestroyDescriptorSetLayout)                      \
    F(vkCreateShaderModule)                              \
    F(vkDestroyShaderModule)                             \
    F(vkCreatePipelineLayout)                            \
    F(vkDestroyPipelineLayout)                           \
    F(vkCreateComputePipelines)                          \
    F(vkDestroyPipeline)                                 \
    F(vkCmdBindPipeline)                                 \
    F(vkCmdPushConstants)                                \
    F(vkCmdDispatch)                                     \
    F(vkCmdPipelineBarrier)                              \
    F(vkCmdCopyBuffer)                                   \
    F(vkCreateQueryPool)                                 \
    F(vkDestroyQueryPool)                                \
    F(vkCmdResetQueryPool)                               \
    F(vkCmdWriteTimestamp)                               \
    F(vkGetQueryPoolResults)

  #define VKOIDN_DECLARE_FUNC(name) PFN_##name name = nullptr;

  struct VulkanFuncs
  {
    VKOIDN_INSTANCE_FUNCS(VKOIDN_DECLARE_FUNC)
    VKOIDN_INSTANCE_FUNCS_OPT(VKOIDN_DECLARE_FUNC)
    VKOIDN_DEVICE_FUNCS(VKOIDN_DECLARE_FUNC)
    PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = nullptr;
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = nullptr; // optional (image views)

    // Loads the functions; returns false if a required function is missing
    bool load(VkInstance instance, VkDevice device,
              PFN_vkGetInstanceProcAddr getInstanceProcAddr,
              PFN_vkGetDeviceProcAddr getDeviceProcAddr)
    {
    #define VKOIDN_LOAD_INSTANCE(name) \
      name = reinterpret_cast<PFN_##name>(getInstanceProcAddr(instance, #name)); if (!name) return false;
    #define VKOIDN_LOAD_INSTANCE_OPT(name) \
      name = reinterpret_cast<PFN_##name>(getInstanceProcAddr(instance, #name));
    #define VKOIDN_LOAD_DEVICE(name) \
      name = reinterpret_cast<PFN_##name>(getDeviceProcAddr(device, #name)); if (!name) return false;

      VKOIDN_INSTANCE_FUNCS(VKOIDN_LOAD_INSTANCE)
      VKOIDN_INSTANCE_FUNCS_OPT(VKOIDN_LOAD_INSTANCE_OPT)
      if (!getDeviceProcAddr)
        getDeviceProcAddr = vkGetDeviceProcAddr;
      VKOIDN_DEVICE_FUNCS(VKOIDN_LOAD_DEVICE)

      vkCmdPushDescriptorSetKHR = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        getDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));

      // Core in Vulkan 1.2, fall back to the KHR alias if needed
      vkGetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
        getDeviceProcAddr(device, "vkGetBufferDeviceAddress"));
      if (!vkGetBufferDeviceAddress)
        vkGetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
          getDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
      return vkGetBufferDeviceAddress != nullptr;

    #undef VKOIDN_LOAD_INSTANCE
    #undef VKOIDN_LOAD_INSTANCE_OPT
    #undef VKOIDN_LOAD_DEVICE
    }
  };

  #undef VKOIDN_DECLARE_FUNC

} // namespace vkoidn
