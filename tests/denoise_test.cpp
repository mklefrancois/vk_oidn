// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Command-line test for vk_oidn: denoises PFM images on a Vulkan device created like an
// application would (using the vk_oidn device requirement helpers), optionally compares the
// output to a reference image and measures the GPU time.

#include <vulkan/vulkan.h>
#include "vk_oidn/vk_oidn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x) \
  do { VkResult r = (x); if (r != VK_SUCCESS) throw std::runtime_error(std::string(#x) + " failed: " + std::to_string(r)); } while (0)

// Number of errors reported by the validation layer (--validation)
static int g_validationErrors = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                    VkDebugUtilsMessageTypeFlagsEXT,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
  const bool error = severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  if (error)
    ++g_validationErrors;
  fprintf(stderr, "[validation %s] %s\n", error ? "error" : "warning", data->pMessage);
  return VK_FALSE;
}

// -------------------------------------------------------------------------------------------------
// Images
// -------------------------------------------------------------------------------------------------

struct Image
{
  int width = 0, height = 0;
  std::vector<float> data; // RGB
};

static Image loadPFM(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot open " + path);
  std::string type;
  float scale;
  Image image;
  file >> type >> image.width >> image.height >> scale;
  file.get();
  const int C = (type == "PF") ? 3 : (type == "Pf" ? 1 : 0);
  if (!C)
    throw std::runtime_error("invalid PFM file " + path);
  std::vector<float> raw(size_t(image.width) * image.height * C);
  file.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float));
  image.data.resize(size_t(image.width) * image.height * 3);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < image.width; ++x)
      for (int c = 0; c < 3; ++c)
        image.data[(size_t(image.height - 1 - y) * image.width + x) * 3 + c] =
          raw[(size_t(y) * image.width + x) * C + (C == 3 ? c : 0)];
  return image;
}

static void savePFM(const std::string& path, const Image& image)
{
  std::ofstream file(path, std::ios::binary);
  file << "PF\n" << image.width << " " << image.height << "\n-1.0\n";
  for (int y = image.height - 1; y >= 0; --y)
    file.write(reinterpret_cast<const char*>(&image.data[size_t(y) * image.width * 3]), image.width * 3 * sizeof(float));
}

// Rounds ties up like the Open Image Denoise apps, to get identical input images
static uint16_t floatToHalf(float f)
{
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int exp = int((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF) // inf/nan
    return uint16_t(sign | 0x7C00 | (mant ? 0x200 : 0));
  if (exp >= 31)
    return uint16_t(sign | 0x7C00);
  if (exp <= 0)
  {
    if (exp < -10)
      return uint16_t(sign);
    mant |= 0x800000;
    const int shift = 14 - exp;
    uint32_t h = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem >= half)
      ++h;
    return uint16_t(sign | h);
  }
  uint32_t h = (uint32_t(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFF;
  if (rem >= 0x1000)
    ++h;
  return uint16_t(sign | h);
}

static float halfToFloat(uint16_t h)
{
  const uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, x;
  if (exp == 0)
  {
    if (mant == 0)
      x = sign;
    else
    {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      x = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  }
  else if (exp == 31)
    x = sign | 0x7F800000 | (mant << 13);
  else
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  float f;
  memcpy(&f, &x, 4);
  return f;
}

// Average relative error (SMAPE) and number of outliers, like oidnDenoise
static double compareImages(const Image& a, const Image& ref, size_t& numOutliers)
{
  double sum = 0;
  numOutliers = 0;
  for (size_t i = 0; i < a.data.size(); ++i)
  {
    const double actual = a.data[i], expect = ref.data[i];
    const double absError = std::abs(expect - actual);
    const double relError = absError / (std::abs(expect) + std::abs(actual) + 0.01);
    if (!(absError <= 0.02 || relError <= 0.05))
      ++numOutliers;
    sum += relError;
  }
  return sum / double(a.data.size());
}

// -------------------------------------------------------------------------------------------------
// Vulkan helpers
// -------------------------------------------------------------------------------------------------

struct Buffer
{
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceAddress address = 0;
  void* mapped = nullptr;
  size_t byteSize = 0;
};

struct App
{
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  std::vector<const char*> extensions;
  VkPhysicalDeviceMemoryProperties memProps{};
  float timestampPeriod = 1.f;

  void init(int deviceIndex, bool validation)
  {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vk_oidn_denoise_test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;

    // Optional Khronos validation layer: errors make the test fail
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* debugExtension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messengerInfo.pfnUserCallback = debugCallback;
    if (validation)
    {
      uint32_t numLayers = 0;
      vkEnumerateInstanceLayerProperties(&numLayers, nullptr);
      std::vector<VkLayerProperties> layers(numLayers);
      vkEnumerateInstanceLayerProperties(&numLayers, layers.data());
      bool found = false;
      for (const VkLayerProperties& l : layers)
        found |= strcmp(l.layerName, layer) == 0;
      if (!found)
        throw std::runtime_error("VK_LAYER_KHRONOS_validation not found (install the Vulkan SDK or set VK_LAYER_PATH)");
      instanceInfo.enabledLayerCount = 1;
      instanceInfo.ppEnabledLayerNames = &layer;
      instanceInfo.enabledExtensionCount = 1;
      instanceInfo.ppEnabledExtensionNames = &debugExtension;
      instanceInfo.pNext = &messengerInfo; // also reports instance creation and destruction
    }
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));
    if (validation)
    {
      auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
      VK_CHECK(createMessenger(instance, &messengerInfo, nullptr, &messenger));
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    if (devices.empty())
      throw std::runtime_error("no Vulkan device");
    if (deviceIndex >= 0)
      physicalDevice = devices.at(deviceIndex);
    else
    {
      physicalDevice = devices[0];
      for (VkPhysicalDevice pd : devices) // prefer a discrete GPU
      {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { physicalDevice = pd; break; }
      }
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    timestampPeriod = props.limits.timestampPeriod;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    while (queueFamily < count && !(families[queueFamily].queueFlags & VK_QUEUE_COMPUTE_BIT))
      ++queueFamily;

    // Extensions and features required by vk_oidn
    vkoidn::getRequiredDeviceExtensions(instance, physicalDevice, vkGetInstanceProcAddr, extensions);

    bool hasCoopMat = false;
    for (const char* e : extensions)
      hasCoopMat |= strcmp(e, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;

    // EXPERIMENTAL: NVIDIA cooperative matrices 2 (used by vk_oidn with VK_OIDN_CM2=1)
    uint32_t numExts = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &numExts, nullptr);
    std::vector<VkExtensionProperties> extProps(numExts);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &numExts, extProps.data());
    bool hasCoopMat2 = false;
    for (const auto& e : extProps)
      hasCoopMat2 |= hasCoopMat && strcmp(e.extensionName, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME) == 0;
    if (hasCoopMat2)
      extensions.push_back(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopMat2Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    coopMat2Features.cooperativeMatrixWorkgroupScope = VK_TRUE;
    coopMat2Features.cooperativeMatrixFlexibleDimensions = VK_TRUE;
    coopMat2Features.cooperativeMatrixTensorAddressing = VK_TRUE;

    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    features.pNext = &features11;
    features11.pNext = &features12;
    features12.pNext = &features13;
    features13.pNext = hasCoopMat ? &coopMatFeatures : nullptr;
    coopMatFeatures.pNext = hasCoopMat2 ? &coopMat2Features : nullptr;
    vkoidn::enableRequiredFeatures(features, features11, features12, features13,
                                   hasCoopMat ? &coopMatFeatures : nullptr);
    features12.hostQueryReset = VK_TRUE;

    const float priority = 1.f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = uint32_t(extensions.size());
    deviceInfo.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
  }

  void destroy()
  {
    if (device)
    {
      vkDestroyCommandPool(device, commandPool, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (messenger)
    {
      auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
      destroyMessenger(instance, messenger, nullptr);
    }
    if (instance)
      vkDestroyInstance(instance, nullptr);
    device = VK_NULL_HANDLE;
    messenger = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
  }

  Buffer createBuffer(size_t byteSize, bool hostVisible)
  {
    Buffer b;
    b.byteSize = byteSize;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = byteSize;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(device, &info, nullptr, &b.buffer));
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, b.buffer, &reqs);
    const VkMemoryPropertyFlags flags = hostVisible
      ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount && type == UINT32_MAX; ++i)
      if ((reqs.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags)
        type = i;
    if (type == UINT32_MAX)
      throw std::runtime_error("no suitable memory type");
    VkMemoryAllocateFlagsInfo allocFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &allocFlags;
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    if (hostVisible)
      VK_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device, &addressInfo);
    return b;
  }

  void destroyBuffer(Buffer& b)
  {
    vkDestroyBuffer(device, b.buffer, nullptr);
    vkFreeMemory(device, b.memory, nullptr);
    b = Buffer();
  }

  struct ImageResource
  {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
  };

  // Storage image (RGBA) usable by vk_oidn
  ImageResource createImage(uint32_t width, uint32_t height, VkFormat format)
  {
    ImageResource res;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &info, nullptr, &res.image));
    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(device, res.image, &reqs);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount && type == UINT32_MAX; ++i)
      if ((reqs.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        type = i;
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &res.memory));
    VK_CHECK(vkBindImageMemory(device, res.image, res.memory, 0));
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = res.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &res.view));
    return res;
  }

  void destroyImage(ImageResource& r)
  {
    vkDestroyImageView(device, r.view, nullptr);
    vkDestroyImage(device, r.image, nullptr);
    vkFreeMemory(device, r.memory, nullptr);
    r = ImageResource();
  }

  static void toGeneralLayout(VkCommandBuffer cmd, VkImage image)
  {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
  }

  VkCommandBuffer beginCommands()
  {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    return cmd;
  }

  void submitAndWait(VkCommandBuffer cmd)
  {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
  }

  static void barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkAccessFlags srcAccess,
                      VkPipelineStageFlags dst, VkAccessFlags dstAccess)
  {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
  }
};

// -------------------------------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------------------------------

static void printUsage()
{
  printf("usage: vk_oidn_denoise_test (--hdr|--ldr) color.pfm [--alb albedo.pfm] [--nrm normal.pfm]\n"
         "                            [--srgb] [--clean_aux] [-q high|balanced|fast] [-t float|half]\n"
         "                            [--is inputScale] [--maxmem MB] [-o output.pfm] [-r ref.pfm]\n"
         "                            [--maxerror e] [-n benchmarkRuns] [-d deviceIndex]\n"
         "                            [--images] (use RGBA storage image views instead of buffers)\n"
         "                            [--validation] (Khronos validation layer, fails on errors)\n");
}

int main(int argc, char* argv[])
{
  App app;
  try
  {
    std::string colorPath, albedoPath, normalPath, outputPath, refPath;
    vkoidn::DenoiserCreateInfo createInfo;
    bool half = false;
    bool useImages = false;
    bool validation = false;
    float inputScale = std::numeric_limits<float>::quiet_NaN();
    double maxError = 0.003;
    int numRuns = 0, deviceIndex = -1;

    for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg); return argv[++i]; };
      if (arg == "--hdr")            { colorPath = next(); createInfo.hdr = true; }
      else if (arg == "--ldr")       { colorPath = next(); createInfo.hdr = false; }
      else if (arg == "--alb")       albedoPath = next();
      else if (arg == "--nrm")       normalPath = next();
      else if (arg == "--srgb")      createInfo.srgb = true;
      else if (arg == "--clean_aux") createInfo.cleanAux = true;
      else if (arg == "-q")
      {
        const std::string q = next();
        createInfo.quality = (q == "fast" || q == "f") ? vkoidn::Quality::Fast :
                             (q == "balanced" || q == "b") ? vkoidn::Quality::Balanced : vkoidn::Quality::High;
      }
      else if (arg == "-t")          half = next() == "half";
      else if (arg == "--is")        inputScale = std::stof(next());
      else if (arg == "--maxmem")    createInfo.maxMemoryMB = uint32_t(std::stoul(next()));
      else if (arg == "-o")          outputPath = next();
      else if (arg == "-r")          refPath = next();
      else if (arg == "--maxerror")  maxError = std::stod(next());
      else if (arg == "-n")          numRuns = std::stoi(next());
      else if (arg == "-d")          deviceIndex = std::stoi(next());
      else if (arg == "--images")    useImages = true;
      else if (arg == "--validation") validation = true;
      else { printUsage(); return 1; }
    }
    if (colorPath.empty()) { printUsage(); return 1; }

    // Load the images
    std::vector<Image> inputs = {loadPFM(colorPath)};
    if (!albedoPath.empty()) inputs.push_back(loadPFM(albedoPath));
    if (!normalPath.empty()) inputs.push_back(loadPFM(normalPath));
    const int W = inputs[0].width, H = inputs[0].height;
    createInfo.width = uint32_t(W);
    createInfo.height = uint32_t(H);
    createInfo.albedo = !albedoPath.empty();
    createInfo.normal = !normalPath.empty();

    // Vulkan
    app.init(deviceIndex, validation);

    vkoidn::ContextCreateInfo contextInfo;
    contextInfo.instance = app.instance;
    contextInfo.physicalDevice = app.physicalDevice;
    contextInfo.device = app.device;
    contextInfo.getInstanceProcAddr = vkGetInstanceProcAddr;
    contextInfo.getDeviceProcAddr = vkGetDeviceProcAddr;
    contextInfo.enabledExtensions = app.extensions.data();
    contextInfo.enabledExtensionCount = uint32_t(app.extensions.size());

    std::unique_ptr<vkoidn::Context> context;
    vkoidn::Result result = vkoidn::Context::create(contextInfo, context);
    if (result != vkoidn::Result::Success)
      throw std::runtime_error(std::string("context creation failed: ") + vkoidn::getResultString(result));

    printf("Device: %s, cooperative matrices: %s\n", context->getDeviceName(),
           context->isCooperativeMatrixEnabled() ? "yes" : "no");

    if (useImages && !context->areImageViewsSupported())
      throw std::runtime_error("image views are not supported (VK_KHR_push_descriptor)");

    // Images are stored in device local buffers (RGB) or storage images (RGBA). The alpha of the
    // output image is initialized to a known value to check that it is preserved.
    const int numChannels = useImages ? 4 : 3;
    const float outputAlpha = 0.25f;
    const size_t pixelSize = (half ? 2 : 4) * numChannels;
    const size_t imageByteSize = size_t(W) * H * pixelSize;
    std::vector<Buffer> imageBuffers;
    std::vector<App::ImageResource> images;
    const VkFormat imageFormat = half ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    for (size_t i = 0; i < inputs.size() + 1; ++i)
    {
      if (useImages)
        images.push_back(app.createImage(uint32_t(W), uint32_t(H), imageFormat));
      else
        imageBuffers.push_back(app.createBuffer(imageByteSize, false));
    }
    Buffer staging = app.createBuffer(imageByteSize * (inputs.size() + 1), true);

    // Fill the staging buffer with the inputs and the initial output (zero RGB, known alpha)
    for (size_t i = 0; i < inputs.size() + 1; ++i)
    {
      char* dst = static_cast<char*>(staging.mapped) + i * imageByteSize;
      for (size_t pixel = 0; pixel < size_t(W) * H; ++pixel)
      {
        for (int c = 0; c < numChannels; ++c)
        {
          const float value = (c == 3) ? outputAlpha : (i < inputs.size() ? inputs[i].data[pixel * 3 + c] : 0.f);
          const size_t j = pixel * numChannels + c;
          if (half)
            reinterpret_cast<uint16_t*>(dst)[j] = floatToHalf(value);
          else
            reinterpret_cast<float*>(dst)[j] = value;
        }
      }
    }

    // Create the denoiser (records the weight upload) and upload the images
    VkCommandBuffer cmd = app.beginCommands();
    std::unique_ptr<vkoidn::Denoiser> denoiser;
    result = vkoidn::Denoiser::create(*context, cmd, createInfo, denoiser);
    if (result != vkoidn::Result::Success)
      throw std::runtime_error(std::string("denoiser creation failed: ") + vkoidn::getResultString(result));
    for (size_t i = 0; i < inputs.size() + 1; ++i)
    {
      if (useImages)
      {
        App::toGeneralLayout(cmd, images[i].image);
        VkBufferImageCopy region{};
        region.bufferOffset = i * imageByteSize;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {uint32_t(W), uint32_t(H), 1};
        vkCmdCopyBufferToImage(cmd, staging.buffer, images[i].image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
      }
      else if (i < inputs.size())
      {
        VkBufferCopy region{i * imageByteSize, 0, imageByteSize};
        vkCmdCopyBuffer(cmd, staging.buffer, imageBuffers[i].buffer, 1, &region);
      }
    }
    app.submitAndWait(cmd);

    printf("Weights: %s, tiles: %u, memory: %.1f MB\n", denoiser->getWeightsName(), denoiser->getTileCount(),
           denoiser->getMemoryUsage() / (1024. * 1024.));

    auto makeImage = [&](const Buffer& b)
    {
      vkoidn::BufferImage image;
      image.address = b.address;
      image.format = half ? vkoidn::Format::Half : vkoidn::Format::Float;
      image.channels = 3;
      return image;
    };

    vkoidn::DenoiseInputs denoiseInputs;
    if (useImages)
    {
      denoiseInputs.colorView.view = images[0].view;
      if (createInfo.albedo) denoiseInputs.albedoView.view = images[1].view;
      if (createInfo.normal) denoiseInputs.normalView.view = images[2].view;
      denoiseInputs.outputView.view = images.back().view;
      denoiseInputs.outputView.format = imageFormat;
    }
    else
    {
      denoiseInputs.color = makeImage(imageBuffers[0]);
      if (createInfo.albedo) denoiseInputs.albedo = makeImage(imageBuffers[1]);
      if (createInfo.normal) denoiseInputs.normal = makeImage(imageBuffers[2]);
      denoiseInputs.output = makeImage(imageBuffers.back());
    }
    denoiseInputs.inputScale = inputScale;

    // Denoise
    cmd = app.beginCommands();
    App::barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    result = denoiser->cmdDenoise(cmd, denoiseInputs);
    if (result != vkoidn::Result::Success)
      throw std::runtime_error(std::string("cmdDenoise failed: ") + vkoidn::getResultString(result));
    App::barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    if (useImages)
    {
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {uint32_t(W), uint32_t(H), 1};
      vkCmdCopyImageToBuffer(cmd, images.back().image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
    }
    else
    {
      VkBufferCopy region{0, 0, imageByteSize};
      vkCmdCopyBuffer(cmd, imageBuffers.back().buffer, staging.buffer, 1, &region);
    }
    App::barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    app.submitAndWait(cmd);

    Image output;
    output.width = W;
    output.height = H;
    output.data.resize(size_t(W) * H * 3);
    auto readValue = [&](size_t j)
    {
      return half ? halfToFloat(static_cast<const uint16_t*>(staging.mapped)[j])
                  : static_cast<const float*>(staging.mapped)[j];
    };
    size_t alphaErrors = 0;
    for (size_t pixel = 0; pixel < size_t(W) * H; ++pixel)
    {
      for (int c = 0; c < 3; ++c)
        output.data[pixel * 3 + c] = readValue(pixel * numChannels + c);
      if (useImages && readValue(pixel * numChannels + 3) != outputAlpha)
        ++alphaErrors;
    }
    if (useImages)
      printf("Output alpha preserved: %s\n", alphaErrors == 0 ? "yes" : "NO");

    // Benchmark
    if (numRuns > 0)
    {
      VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
      queryInfo.queryCount = 2;
      VkQueryPool queryPool;
      VK_CHECK(vkCreateQueryPool(app.device, &queryInfo, nullptr, &queryPool));
      vkResetQueryPool(app.device, queryPool, 0, 2);

      cmd = app.beginCommands();
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
      for (int run = 0; run < numRuns; ++run)
        denoiser->cmdDenoise(cmd, denoiseInputs);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
      app.submitAndWait(cmd);

      uint64_t timestamps[2];
      VK_CHECK(vkGetQueryPoolResults(app.device, queryPool, 0, 2, sizeof(timestamps), timestamps,
                                     sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      const double msec = double(timestamps[1] - timestamps[0]) * app.timestampPeriod * 1e-6 / numRuns;
      printf("GPU time: %.3f msec/image (%d runs)\n", msec, numRuns);
      vkDestroyQueryPool(app.device, queryPool, nullptr);
    }

    for (auto& b : imageBuffers)
      app.destroyBuffer(b);
    for (auto& image : images)
      app.destroyImage(image);
    app.destroyBuffer(staging);
    denoiser.reset();
    context.reset();

    if (!outputPath.empty())
      savePFM(outputPath, output);

    int exitCode = 0;
    if (!refPath.empty())
    {
      const Image ref = loadPFM(refPath);
      if (ref.width != W || ref.height != H)
        throw std::runtime_error("reference size mismatch");
      size_t numOutliers;
      const double avgError = compareImages(output, ref, numOutliers);
      const bool ok = avgError <= maxError && numOutliers == 0;
      printf("Compare: avgerror=%g, outliers=%zu -> %s\n", avgError, numOutliers, ok ? "OK" : "FAILED");
      exitCode = (ok && alphaErrors == 0) ? 0 : 1;
    }

    app.destroy();
    if (validation)
    {
      printf("Validation errors: %d -> %s\n", g_validationErrors, g_validationErrors == 0 ? "OK" : "FAILED");
      if (g_validationErrors != 0)
        exitCode = 1;
    }
    return exitCode;
  }
  catch (const std::exception& e)
  {
    fprintf(stderr, "Error: %s\n", e.what());
    app.destroy();
    return 1;
  }
}
