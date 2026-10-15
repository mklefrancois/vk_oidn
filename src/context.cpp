// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"
#include "kernels.h"
#include "kernel_params.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vkoidn {

  namespace
  {
    bool hasExtension(const ContextCreateInfo& info, const char* name)
    {
      for (uint32_t i = 0; i < info.enabledExtensionCount; ++i)
      {
        if (info.enabledExtensions[i] && strcmp(info.enabledExtensions[i], name) == 0)
          return true;
      }
      return false;
    }

    // Returns the value of an integer environment variable, or defaultValue if not set
    int getEnvInt(const char* name, int defaultValue)
    {
      const char* value = getenv(name);
      return (value && *value) ? atoi(value) : defaultValue;
    }
  }

  Result Context::create(const ContextCreateInfo& createInfo, std::unique_ptr<Context>& context)
  {
    context.reset();
    auto impl = std::make_unique<ContextImpl>();
    const Result result = impl->init(createInfo);
    if (result == Result::Success)
      context = std::move(impl);
    return result;
  }

  ContextImpl::~ContextImpl()
  {
    if (!device)
      return;
    for (auto& p : pipelines)
      fn.vkDestroyPipeline(device, p.second, allocator);
    if (pipelineLayout)
      fn.vkDestroyPipelineLayout(device, pipelineLayout, allocator);
    if (imagePipelineLayout)
      fn.vkDestroyPipelineLayout(device, imagePipelineLayout, allocator);
    if (imageSetLayout)
      fn.vkDestroyDescriptorSetLayout(device, imageSetLayout, allocator);
  }

  Result ContextImpl::init(const ContextCreateInfo& createInfo)
  {
    if (!createInfo.instance || !createInfo.physicalDevice || !createInfo.device ||
        !createInfo.getInstanceProcAddr ||
        (createInfo.enabledExtensionCount > 0 && !createInfo.enabledExtensions))
      return Result::ErrorInvalidArgument;

    if (!fn.load(createInfo.instance, createInfo.device, createInfo.getInstanceProcAddr,
                 createInfo.getDeviceProcAddr))
      return Result::ErrorUnsupportedDevice;

    physicalDevice = createInfo.physicalDevice;
    pipelineCache = createInfo.pipelineCache;
    allocator = createInfo.allocator;

    // Properties
    VkPhysicalDeviceProperties props;
    fn.vkGetPhysicalDeviceProperties(physicalDevice, &props);
    if (props.apiVersion < VK_API_VERSION_1_2 || props.limits.maxPushConstantsSize < maxParamsByteSize)
      return Result::ErrorUnsupportedDevice;
    deviceName = props.deviceName;
    timestampPeriod = props.limits.timestampPeriod;
    const bool isVulkan13 = props.apiVersion >= VK_API_VERSION_1_3;

    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceSubgroupProperties subgroupProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties floatControlsProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    props2.pNext = &subgroupProps;
    subgroupProps.pNext = &floatControlsProps;
    if (isVulkan13)
      floatControlsProps.pNext = &subgroupSizeControlProps;
    fn.vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    subgroupSize = subgroupProps.subgroupSize;
    floatControls16 = floatControlsProps.shaderRoundingModeRTEFloat16 &&
                      floatControlsProps.shaderDenormPreserveFloat16;

    // Subgroup size control is needed only if the subgroup size may vary (enabled by
    // enableRequiredFeatures())
    subgroupSizeControl = isVulkan13 &&
                          subgroupSizeControlProps.minSubgroupSize != subgroupSizeControlProps.maxSubgroupSize &&
                          (subgroupSizeControlProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);

    fn.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    // Cooperative matrices: 16x16x16 FP16 with FP32 accumulation is required, FP16 accumulation
    // is optional. Enabled by default only on NVIDIA GPUs (validated); VK_OIDN_COOPMAT=0/1
    // disables/forces it.
    if (hasExtension(createInfo, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
        fn.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
    {
      uint32_t count = 0;
      fn.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, nullptr);
      std::vector<VkCooperativeMatrixPropertiesKHR> coopMatProps(count, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
      fn.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, coopMatProps.data());

      bool accF32 = false, accF16 = false;
      for (const auto& p : coopMatProps)
      {
        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
            p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            p.scope == VK_SCOPE_SUBGROUP_KHR && !p.saturatingAccumulation)
        {
          if (p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR)
            accF32 = true;
          else if (p.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR)
            accF16 = true;
        }
      }

      const int coopMatOption = getEnvInt("VK_OIDN_COOPMAT", -1);
      const bool isNVIDIA = props.vendorID == 0x10DE;
      coopMat = accF32 && (coopMatOption > 0 || (coopMatOption < 0 && isNVIDIA));
      coopMatAccF16 = coopMat && accF16;
    }

    // EXPERIMENTAL: VK_NV_cooperative_matrix2 convolutions (VK_OIDN_CM2=1); the application must
    // enable the extension with the workgroup scope, flexible dimensions and tensor addressing
    // features
    coopMat2 = coopMat && hasExtension(createInfo, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME) &&
               getEnvInt("VK_OIDN_CM2", 0) > 0 && getEmbeddedKernel("conv_cm2", nullptr) != nullptr;

    // Pipeline layout: push constants only (all memory is accessed through device addresses)
    VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, maxParamsByteSize};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    device = createInfo.device;
    if (fn.vkCreatePipelineLayout(device, &layoutInfo, allocator, &pipelineLayout) != VK_SUCCESS)
      return Result::ErrorVulkan;

    // Image views: storage images in a push descriptor set (color, albedo, normal, output)
    if (hasExtension(createInfo, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) && fn.vkCmdPushDescriptorSetKHR)
    {
      VkDescriptorSetLayoutBinding bindings[4];
      for (uint32_t i = 0; i < 4; ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      VkDescriptorSetLayoutCreateInfo setLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      setLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
      setLayoutInfo.bindingCount = 4;
      setLayoutInfo.pBindings = bindings;
      if (fn.vkCreateDescriptorSetLayout(device, &setLayoutInfo, allocator, &imageSetLayout) != VK_SUCCESS)
        return Result::ErrorVulkan;

      layoutInfo.setLayoutCount = 1;
      layoutInfo.pSetLayouts = &imageSetLayout;
      if (fn.vkCreatePipelineLayout(device, &layoutInfo, allocator, &imagePipelineLayout) != VK_SUCCESS)
        return Result::ErrorVulkan;
    }

    return Result::Success;
  }

  int ContextImpl::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                                  VkMemoryPropertyFlags preferred) const
  {
    for (int pass = 0; pass < 2; ++pass)
    {
      const VkMemoryPropertyFlags flags = (pass == 0) ? (required | preferred) : required;
      for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
      {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags)
          return int(i);
      }
    }
    return -1;
  }

  Result ContextImpl::createBuffer(VkDeviceSize byteSize, bool hostVisible, GpuBuffer& buffer)
  {
    buffer = GpuBuffer();
    buffer.byteSize = byteSize;
    if (byteSize == 0)
      return Result::Success;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = byteSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (fn.vkCreateBuffer(device, &bufferInfo, allocator, &buffer.buffer) != VK_SUCCESS)
      return Result::ErrorVulkan;

    VkMemoryRequirements memReqs;
    fn.vkGetBufferMemoryRequirements(device, buffer.buffer, &memReqs);

    const int memType = hostVisible
      ? findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)
      : findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (memType < 0)
    {
      destroyBuffer(buffer);
      return Result::ErrorOutOfMemory;
    }

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &flagsInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = uint32_t(memType);
    VkResult vr = fn.vkAllocateMemory(device, &allocInfo, allocator, &buffer.memory);
    if (vr != VK_SUCCESS)
    {
      destroyBuffer(buffer);
      return (vr == VK_ERROR_OUT_OF_DEVICE_MEMORY || vr == VK_ERROR_OUT_OF_HOST_MEMORY)
             ? Result::ErrorOutOfMemory : Result::ErrorVulkan;
    }

    if (fn.vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) != VK_SUCCESS ||
        (hostVisible && fn.vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS))
    {
      destroyBuffer(buffer);
      return Result::ErrorVulkan;
    }

    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer.buffer;
    buffer.address = fn.vkGetBufferDeviceAddress(device, &addressInfo);
    return Result::Success;
  }

  void ContextImpl::destroyBuffer(GpuBuffer& buffer)
  {
    if (buffer.buffer)
      fn.vkDestroyBuffer(device, buffer.buffer, allocator);
    if (buffer.memory)
    {
      if (buffer.mapped)
        fn.vkUnmapMemory(device, buffer.memory);
      fn.vkFreeMemory(device, buffer.memory, allocator);
    }
    buffer = GpuBuffer();
  }

  Result ContextImpl::getPipeline(const std::string& kernelName, const std::vector<uint32_t>& specConstants,
                                  bool requireFullSubgroups, VkPipeline& pipeline, bool imageIO)
  {
    if (imageIO && !imagePipelineLayout)
      return Result::ErrorUnsupportedDevice;

    // Use the variant without float controls if the device does not support them
    const std::string moduleName = floatControls16 ? kernelName : kernelName + "_nofc";

    std::string key = moduleName;
    for (uint32_t c : specConstants)
      key += "_" + std::to_string(c);

    std::lock_guard<std::mutex> lock(pipelineMutex);
    auto it = pipelines.find(key);
    if (it != pipelines.end())
    {
      pipeline = it->second;
      return Result::Success;
    }

    size_t codeSize = 0;
    const void* code = getEmbeddedKernel(moduleName.c_str(), &codeSize);

    // Development: VK_OIDN_KERNEL_DIR=<dir> loads <dir>/<kernel>.spv instead of the embedded kernel
    // when the file exists (to try kernel changes without rebuilding the library)
    std::vector<uint32_t> fileCode;
    if (const char* dir = getenv("VK_OIDN_KERNEL_DIR"))
    {
      if (FILE* file = fopen((std::string(dir) + "/" + moduleName + ".spv").c_str(), "rb"))
      {
        fseek(file, 0, SEEK_END);
        const long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        fileCode.resize(size_t(size) / 4);
        if (fread(fileCode.data(), 4, fileCode.size(), file) == fileCode.size())
        {
          code = fileCode.data();
          codeSize = fileCode.size() * 4;
        }
        fclose(file);
      }
    }
    if (!code)
      return Result::ErrorInvalidArgument;

    // The embedded data is 256-byte aligned, which satisfies the alignment of pCode
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = codeSize;
    moduleInfo.pCode = static_cast<const uint32_t*>(code);
    VkShaderModule module;
    if (fn.vkCreateShaderModule(device, &moduleInfo, allocator, &module) != VK_SUCCESS)
      return Result::ErrorVulkan;

    std::vector<VkSpecializationMapEntry> entries(specConstants.size());
    for (size_t i = 0; i < specConstants.size(); ++i)
      entries[i] = {uint32_t(i), uint32_t(i * sizeof(uint32_t)), sizeof(uint32_t)};
    VkSpecializationInfo specInfo{uint32_t(entries.size()), entries.data(),
                                  specConstants.size() * sizeof(uint32_t), specConstants.data()};

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    subgroupSizeInfo.requiredSubgroupSize = subgroupSize;

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.stage.pSpecializationInfo = specConstants.empty() ? nullptr : &specInfo;
    if (requireFullSubgroups && subgroupSizeControl)
    {
      pipelineInfo.stage.pNext = &subgroupSizeInfo;
      pipelineInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    pipelineInfo.layout = imageIO ? imagePipelineLayout : pipelineLayout;

    const VkResult vr = fn.vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, allocator, &pipeline);
    fn.vkDestroyShaderModule(device, module, allocator);
    if (vr != VK_SUCCESS)
      return Result::ErrorVulkan;

    pipelines[key] = pipeline;
    return Result::Success;
  }

  void ContextImpl::cmdComputeBarrier(VkCommandBuffer cmd) const
  {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  void ContextImpl::cmdPushImageViews(VkCommandBuffer cmd, const VkImageView views[4]) const
  {
    VkDescriptorImageInfo imageInfos[4];
    VkWriteDescriptorSet writes[4];
    for (uint32_t i = 0; i < 4; ++i)
    {
      imageInfos[i] = {VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL};
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[i].pImageInfo = &imageInfos[i];
    }
    fn.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imagePipelineLayout, 0, 4, writes);
  }

  void ContextImpl::cmdDispatch(VkCommandBuffer cmd, VkPipeline pipeline, const void* params, uint32_t paramsByteSize,
                                uint32_t numGroupsX, uint32_t numGroupsY, uint32_t numGroupsZ,
                                bool imageIO) const
  {
    fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    fn.vkCmdPushConstants(cmd, imageIO ? imagePipelineLayout : pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, paramsByteSize, params);
    fn.vkCmdDispatch(cmd, numGroupsX, numGroupsY, numGroupsZ);
    cmdComputeBarrier(cmd);
  }

} // namespace vkoidn
