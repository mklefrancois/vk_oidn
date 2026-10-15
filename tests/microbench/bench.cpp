// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Micro-benchmark harness: runs a SPIR-V compute kernel and measures its GPU time.
//
//   bench --info
//   bench <kernel.spv> <groupsX> <groupsY> <groupsZ> [--buf MB]... [--int v]... [--iters n]
//         [--coopvec-matrix bufIndex rows cols count]   (row-major f16 -> InferencingOptimal)
//
// Push constants: { uint64_t buf[4]; int32_t ints[8]; } (96 bytes)
// Buffers are filled with random FP16 values in [-0.5, 0.5].

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x) do { VkResult res_ = (x); if (res_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, res_); exit(1); } } while (0)

static VkInstance instance;
static VkPhysicalDevice phys;
static VkDevice device;
static VkQueue queue;
static uint32_t family = 0;
static VkPhysicalDeviceMemoryProperties memProps;

static uint16_t toHalf(float f)
{
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int exp = int((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (x >> 13) & 0x3FF;
  if (exp <= 0) return uint16_t(sign);
  return uint16_t(sign | (exp << 10) | mant);
}

struct Buffer { VkBuffer buf; VkDeviceMemory mem; VkDeviceAddress addr; size_t size; };

static uint32_t findMemory(uint32_t bits, VkMemoryPropertyFlags flags, VkMemoryPropertyFlags avoid = 0)
{
  for (int pass = 0; pass < 2; ++pass) // first without the avoided flags (e.g. host memory not in the BAR window)
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags &&
          (pass == 1 || !(memProps.memoryTypes[i].propertyFlags & avoid)))
        return i;
  throw std::runtime_error("no memory type");
}

static void destroyBuffer(Buffer& b)
{
  vkDestroyBuffer(device, b.buf, nullptr);
  vkFreeMemory(device, b.mem, nullptr);
  b = {};
}

static Buffer createBuffer(size_t size, bool host)
{
  Buffer b{}; b.size = size;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VK_CHECK(vkCreateBuffer(device, &bi, nullptr, &b.buf));
  VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, b.buf, &req);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = host ? findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                            : findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &b.mem));
  VK_CHECK(vkBindBufferMemory(device, b.buf, b.mem, 0));
  VkBufferDeviceAddressInfo ad{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; ad.buffer = b.buf;
  b.addr = vkGetBufferDeviceAddress(device, &ad);
  return b;
}

static VkCommandPool pool;
static VkCommandBuffer beginCmd()
{
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
  VkCommandBuffer cmd; VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
  return cmd;
}
static void submit(VkCommandBuffer cmd)
{
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(queue));
  vkFreeCommandBuffers(device, pool, 1, &cmd);
}
static void barrier(VkCommandBuffer cmd)
{
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

static const char* typeName(int t)
{
  switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    case 1000141000: return "bf16";
    case 1000491002: return "e4m3";
    case 1000491003: return "e5m2";
    case 1000491000: return "s8packed";
    case 1000491001: return "u8packed";
    default: { static char b[32]; snprintf(b, 32, "t%d", t); return b; }
  }
}

int main(int argc, char** argv)
{
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_4;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
  VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
  uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
  std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(instance, &n, devs.data());
  for (auto d : devs) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p); if (p.vendorID == 0x10DE) { phys = d; break; } }
  if (!phys) { fprintf(stderr, "no NVIDIA GPU\n"); return 1; }
  vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);

  const bool info = argc > 1 && strcmp(argv[1], "--info") == 0;
  if (info)
  {
    printf("%s\n", props.deviceName);
    VkPhysicalDeviceCooperativeVectorPropertiesNV cvp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &cvp};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    printf("coopvec: maxComponents=%u stages=0x%x trainingF16Acc=%d trainingF32Acc=%d\n", cvp.maxCooperativeVectorComponents,
           cvp.cooperativeVectorSupportedStages, cvp.cooperativeVectorTrainingFloat16Accumulation, cvp.cooperativeVectorTrainingFloat32Accumulation);
    auto getCV = (PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeVectorPropertiesNV");
    uint32_t c = 0; getCV(phys, &c, nullptr);
    std::vector<VkCooperativeVectorPropertiesNV> cv(c, {VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV});
    getCV(phys, &c, cv.data());
    for (auto& p : cv)
      printf("  coopvec in=%s(%s) matrix=%s bias=%s result=%s transpose=%d\n", typeName(p.inputType), typeName(p.inputInterpretation),
             typeName(p.matrixInterpretation), typeName(p.biasInterpretation), typeName(p.resultType), p.transpose);
    auto getCM = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    getCM(phys, &c, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> cm(c, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    getCM(phys, &c, cm.data());
    for (auto& p : cm)
      printf("  coopmat %ux%ux%u A=%s B=%s C=%s R=%s scope=%d\n", p.MSize, p.NSize, p.KSize, typeName(p.AType), typeName(p.BType),
             typeName(p.CType), typeName(p.ResultType), p.scope);
    return 0;
  }
  if (argc < 5) { fprintf(stderr, "usage: bench <kernel.spv> gx gy gz [--buf MB]... [--int v]... [--iters n] [--coopvec-matrix buf rows cols count]\n"); return 1; }

  // Device with the features used by the kernels
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceCooperativeVectorFeaturesNV fcv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV};
  VkPhysicalDeviceShaderFloat8FeaturesEXT ff8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
  f2.pNext = &f11; f11.pNext = &f12; f12.pNext = &f13; f13.pNext = &fcm; fcm.pNext = &fcv; fcv.pNext = &ff8;
  vkGetPhysicalDeviceFeatures2(phys, &f2); // enable everything supported
  const char* exts[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME};
  const float prio = 1.f;
  VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = 0; qi.queueCount = 1; qi.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qi; dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = exts;
  VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
  vkGetDeviceQueue(device, 0, 0, &queue);
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = 0;
  VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));

  // Arguments
  const std::string spvPath = argv[1];
  const uint32_t gx = atoi(argv[2]), gy = atoi(argv[3]), gz = atoi(argv[4]);
  std::vector<double> bufMB; std::vector<int> ints; int iters = 20;
  struct CvMatrix { int buf, rows, cols, count; }; std::vector<CvMatrix> cvMatrices;
  for (int i = 5; i < argc; ++i)
  {
    std::string a = argv[i];
    if (a == "--buf") bufMB.push_back(atof(argv[++i]));
    else if (a == "--int") ints.push_back(atoi(argv[++i]));
    else if (a == "--iters") iters = atoi(argv[++i]);
    else if (a == "--coopvec-matrix") { CvMatrix m; m.buf = atoi(argv[++i]); m.rows = atoi(argv[++i]); m.cols = atoi(argv[++i]); m.count = atoi(argv[++i]); cvMatrices.push_back(m); }
  }

  // Buffers with random FP16 data
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<Buffer> bufs;
  for (double mb : bufMB)
  {
    const size_t size = std::max<size_t>(256, size_t(mb * 1048576.0) / 256 * 256);
    Buffer b = createBuffer(size, false);
    Buffer s = createBuffer(size, true);
    void* p; vkMapMemory(device, s.mem, 0, size, 0, &p);
    uint16_t* h = (uint16_t*)p;
    for (size_t j = 0; j < size / 2; ++j) h[j] = toHalf(dist(rng));
    vkUnmapMemory(device, s.mem);
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{0, 0, size}; vkCmdCopyBuffer(cmd, s.buf, b.buf, 1, &region);
    submit(cmd);
    destroyBuffer(s);
    bufs.push_back(b);
  }

  // Convert row-major FP16 matrices to the cooperative vector inferencing-optimal layout (replaces the buffer)
  for (const CvMatrix& m : cvMatrices)
  {
    auto convert = (PFN_vkConvertCooperativeVectorMatrixNV)vkGetDeviceProcAddr(device, "vkConvertCooperativeVectorMatrixNV");
    auto cmdConvert = (PFN_vkCmdConvertCooperativeVectorMatrixNV)vkGetDeviceProcAddr(device, "vkCmdConvertCooperativeVectorMatrixNV");
    VkConvertCooperativeVectorMatrixInfoNV ci{VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV};
    size_t dstSize = 0;
    ci.srcSize = size_t(m.rows) * m.cols * 2;
    ci.pDstSize = &dstSize;
    ci.srcComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    ci.dstComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    ci.numRows = m.rows; ci.numColumns = m.cols;
    ci.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV; ci.srcStride = m.cols * 2;
    ci.dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV; ci.dstStride = 0;
    VK_CHECK(convert(device, &ci));
    const size_t stride = (dstSize + 63) / 64 * 64;
    printf("coopvec matrix %dx%d: optimal size %zu bytes (row-major %zu)\n", m.rows, m.cols, dstSize, size_t(ci.srcSize));
    Buffer dst = createBuffer(stride * m.count, false);
    VkCommandBuffer cmd = beginCmd();
    std::vector<VkConvertCooperativeVectorMatrixInfoNV> infos(m.count, ci);
    std::vector<size_t> sizes(m.count, dstSize);
    for (int k = 0; k < m.count; ++k)
    {
      infos[k].srcData.deviceAddress = bufs[m.buf].addr + size_t(k) * ci.srcSize;
      infos[k].dstData.deviceAddress = dst.addr + size_t(k) * stride;
      infos[k].pDstSize = &sizes[k];
    }
    cmdConvert(cmd, m.count, infos.data());
    submit(cmd);
    bufs[m.buf] = dst;
    ints.push_back(int(stride)); // stride between converted matrices, appended to the ints
  }

  // Pipeline
  std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", spvPath.c_str()); return 1; }
  std::vector<char> code(size_t(f.tellg())); f.seekg(0); f.read(code.data(), code.size());
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t*)code.data();
  VkShaderModule sm; VK_CHECK(vkCreateShaderModule(device, &smi, nullptr, &sm));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 96};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
  VkPipelineLayout layout; VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &layout));
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"};
  cpi.layout = layout;
  VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

  struct { uint64_t buf[4]; int32_t ints[8]; } pc{};
  for (size_t i = 0; i < bufs.size() && i < 4; ++i) pc.buf[i] = bufs[i].addr;
  for (size_t i = 0; i < ints.size() && i < 8; ++i) pc.ints[i] = ints[i];

  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
  VkQueryPool qp; VK_CHECK(vkCreateQueryPool(device, &qpi, nullptr, &qp));
  std::vector<double> times;
  for (int it = 0; it < iters + 2; ++it)
  {
    VkCommandBuffer cmd = beginCmd();
    vkCmdResetQueryPool(cmd, qp, 0, 2);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    barrier(cmd);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    vkCmdDispatch(cmd, gx, gy, gz);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    submit(cmd);
    uint64_t ts[2]; VK_CHECK(vkGetQueryPoolResults(device, qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    if (it >= 2) times.push_back(double(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6);
  }
  std::sort(times.begin(), times.end());
  printf("time: median %.4f ms, min %.4f ms\n", times[times.size() / 2], times[0]);
  return 0;
}
