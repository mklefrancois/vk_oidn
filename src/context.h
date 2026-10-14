// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vk_oidn/vk_oidn.h"
#include "vulkan_funcs.h"
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkoidn {

  // Buffer with its own memory allocation and device address
  struct GpuBuffer
  {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    void* mapped = nullptr; // host pointer if host visible
    VkDeviceSize byteSize = 0;
  };

  class ContextImpl final : public Context
  {
  public:
    ~ContextImpl() override;

    Result init(const ContextCreateInfo& createInfo);

    bool isCooperativeMatrixEnabled() const override { return coopMat; }
    const char* getDeviceName() const override { return deviceName.c_str(); }
    bool areImageViewsSupported() const override { return imagePipelineLayout != VK_NULL_HANDLE; }

    const VulkanFuncs& getFuncs() const { return fn; }
    VkDevice getDevice() const { return device; }
    bool hasCoopMatAccF16() const { return coopMatAccF16; }
    uint32_t getSubgroupSize() const { return subgroupSize; }

    // Memory
    Result createBuffer(VkDeviceSize byteSize, bool hostVisible, GpuBuffer& buffer);
    void destroyBuffer(GpuBuffer& buffer);

    // Returns a compute pipeline (cached) for the kernel with the specified name and
    // specialization constants (constant_id = index)
    // Kernels using image views (imageIO) use a pipeline layout with a push descriptor set.
    Result getPipeline(const std::string& kernelName, const std::vector<uint32_t>& specConstants,
                       bool requireFullSubgroups, VkPipeline& pipeline, bool imageIO = false);

    // Records the push descriptors of the image views (color, albedo, normal, output) for the
    // kernels using image views
    void cmdPushImageViews(VkCommandBuffer cmd, const VkImageView views[4]) const;

    // Records a dispatch followed by a compute -> compute memory barrier
    void cmdDispatch(VkCommandBuffer cmd, VkPipeline pipeline, const void* params, uint32_t paramsByteSize,
                     uint32_t numGroupsX, uint32_t numGroupsY, uint32_t numGroupsZ,
                     bool imageIO = false) const;

    template<typename Params>
    void cmdDispatch(VkCommandBuffer cmd, VkPipeline pipeline, const Params& params,
                     uint32_t numGroupsX, uint32_t numGroupsY, uint32_t numGroupsZ,
                     bool imageIO = false) const
    {
      cmdDispatch(cmd, pipeline, &params, uint32_t(sizeof(Params)), numGroupsX, numGroupsY, numGroupsZ, imageIO);
    }

    void cmdComputeBarrier(VkCommandBuffer cmd) const;

  private:
    int findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) const;

    VulkanFuncs fn;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    const VkAllocationCallbacks* allocator = nullptr;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout imageSetLayout = VK_NULL_HANDLE; // push descriptors (image views)
    VkPipelineLayout imagePipelineLayout = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    std::string deviceName;

    // Capabilities
    bool coopMat = false;
    bool coopMatAccF16 = false;
    bool floatControls16 = false;
    bool subgroupSizeControl = false;
    uint32_t subgroupSize = 32;

    std::mutex pipelineMutex;
    std::unordered_map<std::string, VkPipeline> pipelines;
  };

} // namespace vkoidn
