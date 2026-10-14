// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"
#include "kernel_params.h"
#include "weights.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vkoidn {

  namespace
  {
    constexpr int minTileAlignment = 16;           // spatial alignment of the tiles in pixels
    constexpr int defaultMaxTileSize = 2160 * 2160; // default maximum number of pixels per tile
    constexpr size_t memoryAlignment = 256;

    template<typename T> T ceilDiv(T a, T b) { return (a + b - 1) / b; }
    template<typename T> T roundUp(T a, T b) { return ceilDiv(a, b) * b; }

    // Smallest integer >= a with remainder c when divided by b
    int roundUp(int a, int b, int c) { return (a + b - c - 1) / b * b + c; }

    // Transfer functions (must match the shaders)
    enum class Transfer { Linear = 0, SRGB = 1, PU = 2 };

    float puForward(float y)
    {
      const float a = 1.41283765e+03f, b = 1.64593172e+00f, c = 4.31384981e-01f, d = -2.94139609e-03f;
      const float e = 1.92653254e-01f, f = 6.26026094e-03f, g = 9.98620152e-01f;
      const float y0 = 1.57945760e-06f, y1 = 3.22087631e-02f;
      if (y <= y0)
        return a * y;
      else if (y <= y1)
        return b * std::pow(y, c) + d;
      else
        return e * std::log(y + f) + g;
    }

    float srgbForward(float y)
    {
      return (y <= 0.0031308f) ? (12.92f * y) : (1.055f * std::pow(y, 1.f / 2.4f) - 0.055f);
    }

    ImageParams getImageParams(const BufferImage& image, uint32_t width)
    {
      const uint32_t elementSize = (image.format == Format::Half) ? 2 : 4;
      const uint32_t pixelByteStride = image.pixelByteStride ? image.pixelByteStride : image.channels * elementSize;
      const uint32_t rowByteStride = image.rowByteStride ? image.rowByteStride : width * pixelByteStride;

      ImageParams params;
      params.ptr = image.address;
      params.hByteStride = rowByteStride;
      params.fmt = pixelByteStride | (image.channels << 24) | (image.format == Format::Half ? (1u << 28) : 0u);
      return params;
    }

    bool isHalfFormat(VkFormat format)
    {
      return format == VK_FORMAT_R16G16B16A16_SFLOAT || format == VK_FORMAT_R16G16B16_SFLOAT ||
             format == VK_FORMAT_R16G16_SFLOAT || format == VK_FORMAT_R16_SFLOAT;
    }

    bool isValidImage(const BufferImage& image)
    {
      const uint32_t elementSize = (image.format == Format::Half) ? 2 : 4;
      return image.address != 0 && image.channels >= 1 && image.channels <= 3 &&
             image.pixelByteStride < (1u << 24) &&
             (image.pixelByteStride == 0 || image.pixelByteStride >= image.channels * elementSize) &&
             (image.address % elementSize) == 0;
    }
  }

  // -----------------------------------------------------------------------------------------------
  // DenoiserImpl
  // -----------------------------------------------------------------------------------------------

  class DenoiserImpl final : public Denoiser
  {
  public:
    explicit DenoiserImpl(ContextImpl& context) : ctx(context) {}
    ~DenoiserImpl() override;

    Result init(VkCommandBuffer cmd, const DenoiserCreateInfo& createInfo);

    Result cmdDenoise(VkCommandBuffer cmd, const DenoiseInputs& inputs) override;
    uint64_t getMemoryUsage() const override;
    uint32_t getTileCount() const override { return uint32_t(tileCountH * tileCountW); }
    const char* getWeightsName() const override { return weightsName.c_str(); }

  private:
    // Tensor: f16 HWC with padded channels, allocated in the scratch buffer
    struct Tensor
    {
      int C = 0, H = 0, W = 0;
      size_t byteSize = 0;
      size_t offset = 0;
      int first = 0; // index of the producer (-1 = input processing)
      int last = 0;  // index of the last consumer (numLayers = output processing)
    };

    struct Layer
    {
      WeightsLayer desc;
      int convH = 0, convW = 0;    // convolution resolution
      int src0 = 0, src1 = -1;     // tensor indices
      int dst = 0;                 // tensor index
      int tileW = 0, tileH = 0;    // kernel workgroup tile
      int numFrags = 1;            // output blocks per workgroup (cooperative matrices)
      VkPipeline pipeline = VK_NULL_HANDLE;

      // Experimental VK_NV_cooperative_matrix2 kernel
      bool cm2 = false;
      int cm2BN = 0;               // output channels per workgroup
      uint64_t cm2WeightOffset = 0; // [tap][K][N] f16 weights in cm2Buffer
      uint64_t cm2BiasOffset = 0;   // [N] f32 bias in cm2Buffer
    };

    // Computes the tensor shapes and the scratch memory plan for the current tile size
    size_t planMemory();
    Result createPipelines(bool accF16);
    void cmdAutoexposure(VkCommandBuffer cmd, const ImageParams& color, uint64_t resultAddress,
                         const VkImageView* views);

    ContextImpl& ctx;
    DenoiserCreateInfo info;
    std::string weightsName;
    std::vector<uint8_t> weightsBlob; // copy of the packed weights (header + layers + data)
    Weights weights;

    Transfer transfer = Transfer::Linear;
    float normScale = 1.f, rcpNormScale = 1.f;
    bool accF16 = false; // FP16 accumulation in the convolutions

    // Tiling
    int H = 0, W = 0;
    int tileH = 0, tileW = 0, tilePadH = 0, tilePadW = 0;
    int tileCountH = 1, tileCountW = 1;
    int tileOverlap = 0;

    std::vector<Tensor> tensors; // 0 = network input, i + 1 = output of layer i
    std::vector<Layer> layers;
    size_t scratchByteSize = 0;

    // Autoexposure
    int aeNumBinsH = 0, aeNumBinsW = 0, aeNumGroups = 0;
    size_t aeByteSize = 0;

    // Pipelines
    VkPipeline inputPipeline = VK_NULL_HANDLE;
    VkPipeline outputPipeline = VK_NULL_HANDLE;
    VkPipeline aeDownsamplePipeline = VK_NULL_HANDLE;
    VkPipeline aeReducePipeline = VK_NULL_HANDLE;
    VkPipeline aeReduceFinalPipeline = VK_NULL_HANDLE;

    // Pipelines for image view inputs/outputs (if supported)
    VkPipeline inputImagePipeline = VK_NULL_HANDLE;
    VkPipeline outputImagePipeline = VK_NULL_HANDLE;
    VkPipeline aeDownsampleImagePipeline = VK_NULL_HANDLE;

    // GPU memory
    GpuBuffer weightBuffer;
    GpuBuffer stagingBuffer; // weights upload
    GpuBuffer scratchBuffer; // tensors
    GpuBuffer aeBuffer;      // autoexposure bins, partial sums and result
    GpuBuffer cm2Buffer;     // weights for the experimental coopmat2 kernel
    GpuBuffer cm2Staging;

    // Profiling (VK_OIDN_PROFILE=1): GPU time of each dispatch of the first tile, averaged over
    // the executions and printed when the denoiser is destroyed
    void cmdProfile(VkCommandBuffer cmd, uint32_t query);
    void collectProfile();
    VkQueryPool queryPool = VK_NULL_HANDLE;
    uint32_t numQueries = 0;
    bool profilePending = false;
    bool profiling = false; // recording the first tile
    std::vector<std::string> profileNames;
    std::vector<double> profileTimes; // accumulated msec per entry
    int profileCount = 0;
  };

  Result Denoiser::create(Context& context, VkCommandBuffer cmd, const DenoiserCreateInfo& createInfo,
                          std::unique_ptr<Denoiser>& denoiser)
  {
    denoiser.reset();
    auto impl = std::make_unique<DenoiserImpl>(static_cast<ContextImpl&>(context));
    const Result result = impl->init(cmd, createInfo);
    if (result == Result::Success)
      denoiser = std::move(impl);
    return result;
  }

  DenoiserImpl::~DenoiserImpl()
  {
    if (queryPool)
    {
      collectProfile();
      if (profileCount > 0)
      {
        double total = 0;
        printf("vk_oidn profile (%s, %dx%d, %d run(s), first tile):\n", weightsName.c_str(), tileW, tileH, profileCount);
        for (size_t i = 0; i < profileNames.size(); ++i)
        {
          printf("  %-14s %8.3f ms\n", profileNames[i].c_str(), profileTimes[i] / profileCount);
          total += profileTimes[i] / profileCount;
        }
        printf("  %-14s %8.3f ms\n", "total", total);
      }
      ctx.getFuncs().vkDestroyQueryPool(ctx.getDevice(), queryPool, ctx.getAllocator());
    }
    ctx.destroyBuffer(weightBuffer);
    ctx.destroyBuffer(stagingBuffer);
    ctx.destroyBuffer(scratchBuffer);
    ctx.destroyBuffer(aeBuffer);
    ctx.destroyBuffer(cm2Buffer);
    ctx.destroyBuffer(cm2Staging);
  }

  uint64_t DenoiserImpl::getMemoryUsage() const
  {
    return weightBuffer.byteSize + scratchBuffer.byteSize + aeBuffer.byteSize + cm2Buffer.byteSize;
  }

  size_t DenoiserImpl::planMemory()
  {
    const int numLayers = int(weights.header->numLayers);
    tensors.assign(numLayers + 1, Tensor());
    layers.assign(numLayers, Layer());

    // Network input
    tensors[0].C = int(roundUp(weights.header->inputChannels, 16u));
    tensors[0].H = tileH;
    tensors[0].W = tileW;
    tensors[0].first = -1;
    tensors[0].last = -1;

    for (int i = 0; i < numLayers; ++i)
    {
      Layer& layer = layers[i];
      layer.desc = weights.layers[i];
      layer.src0 = layer.desc.src0 + 1; // weightsSrcInput (-1) -> tensor 0
      layer.src1 = (layer.desc.src1 == weightsSrcNone) ? -1 : layer.desc.src1 + 1;
      layer.dst = i + 1;

      const Tensor& src0 = tensors[layer.src0];
      const bool upsample = (layer.desc.flags & weightsLayerUpsampleSrc0) != 0;
      const bool pool = (layer.desc.flags & weightsLayerPoolDst) != 0;
      layer.convH = upsample ? src0.H * 2 : src0.H;
      layer.convW = upsample ? src0.W * 2 : src0.W;

      Tensor& dst = tensors[layer.dst];
      dst.C = int(layer.desc.outChannels);
      dst.H = pool ? layer.convH / 2 : layer.convH;
      dst.W = pool ? layer.convW / 2 : layer.convW;
      dst.first = i;
      dst.last = i;

      tensors[layer.src0].last = std::max(tensors[layer.src0].last, i);
      if (layer.src1 >= 0)
        tensors[layer.src1].last = std::max(tensors[layer.src1].last, i);
    }
    tensors[numLayers].last = numLayers; // read by the output processing

    // Assign offsets: largest tensors first, at the lowest offset not overlapping in memory with
    // any already placed tensor whose lifetime overlaps
    std::vector<int> order(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i)
    {
      tensors[i].byteSize = roundUp(size_t(tensors[i].C) * tensors[i].H * tensors[i].W * 2, memoryAlignment);
      order[i] = int(i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return tensors[a].byteSize > tensors[b].byteSize; });

    std::vector<int> placed;
    size_t totalByteSize = 0;
    for (int t : order)
    {
      Tensor& tensor = tensors[t];

      // Collect the memory ranges of the live placed tensors, sorted by offset
      std::vector<std::pair<size_t, size_t>> ranges;
      for (int p : placed)
      {
        const Tensor& other = tensors[p];
        if (other.first <= tensor.last && tensor.first <= other.last)
          ranges.push_back({other.offset, other.offset + other.byteSize});
      }
      std::sort(ranges.begin(), ranges.end());

      size_t offset = 0;
      for (const auto& r : ranges)
      {
        if (offset + tensor.byteSize <= r.first)
          break;
        offset = std::max(offset, r.second);
      }

      tensor.offset = offset;
      totalByteSize = std::max(totalByteSize, offset + tensor.byteSize);
      placed.push_back(t);
    }

    return totalByteSize;
  }

  Result DenoiserImpl::createPipelines(bool accF16)
  {
    Result result;
    if ((result = ctx.getPipeline("input_process", {}, false, inputPipeline)) != Result::Success ||
        (result = ctx.getPipeline("output_process", {}, false, outputPipeline)) != Result::Success)
      return result;

    if (ctx.areImageViewsSupported())
    {
      if ((result = ctx.getPipeline("input_process_image", {}, false, inputImagePipeline, true)) != Result::Success ||
          (result = ctx.getPipeline("output_process_image", {}, false, outputImagePipeline, true)) != Result::Success ||
          (info.hdr &&
           (result = ctx.getPipeline("autoexposure_downsample_image", {}, false, aeDownsampleImagePipeline, true)) != Result::Success))
        return result;
    }

    if (info.hdr)
    {
      if ((result = ctx.getPipeline("autoexposure_downsample", {}, false, aeDownsamplePipeline)) != Result::Success ||
          (result = ctx.getPipeline("autoexposure_reduce", {}, false, aeReducePipeline)) != Result::Success ||
          (result = ctx.getPipeline("autoexposure_reduce_final", {}, false, aeReduceFinalPipeline)) != Result::Success)
        return result;
    }

    for (Layer& layer : layers)
    {
      const uint32_t upsample = (layer.desc.flags & weightsLayerUpsampleSrc0) ? 1 : 0;
      const uint32_t pool     = (layer.desc.flags & weightsLayerPoolDst) ? 1 : 0;
      const uint32_t relu     = (layer.desc.flags & weightsLayerReLU) ? 1 : 0;
      const int numBlocks = int(layer.desc.outChannels / 16);
      const int numInputChannels = int(layer.desc.inChannels0 + layer.desc.inChannels1);

      if (ctx.useCoopMat2() && !upsample && !pool && !accF16)
      {
        // EXPERIMENTAL: one workgroup-scope matrix per BM pixels of a row
        const int N = numBlocks * 16;
        const int numSplits = ceilDiv(N, 128);
        const int BN = roundUp(ceilDiv(N, numSplits), 16);
        const int BM = getenv("VK_OIDN_CM2_BM") ? atoi(getenv("VK_OIDN_CM2_BM")) : 64;
        int BK = (layer.desc.inChannels0 % 32 == 0 && layer.desc.inChannels1 % 32 == 0) ? 32 : 16;
        if (getenv("VK_OIDN_CM2_BK"))
        {
          const int requestedBK = atoi(getenv("VK_OIDN_CM2_BK"));
          if (layer.desc.inChannels0 % requestedBK == 0 && layer.desc.inChannels1 % requestedBK == 0)
            BK = requestedBK;
        }
        layer.cm2 = true;
        layer.cm2BN = BN;
        layer.tileW = BM;
        layer.tileH = 1;
        layer.numFrags = BN / 16;
        if ((result = ctx.getPipeline("conv_cm2", {128, 0, 0, relu, uint32_t(BM), uint32_t(BN), uint32_t(BK)},
                                      false, layer.pipeline)) != Result::Success)
          return result;
      }
      else if (ctx.isCooperativeMatrixEnabled())
      {
        // Select the tile configuration (tuned on NVIDIA Blackwell):
        // - 1 output block per workgroup if the number of output blocks is odd (no wasted work)
        //   or if there would not be enough workgroups to fill the GPU
        // - wider tiles for layers with many input channels (more reuse of the weights)
        int mx = 2, nf = 2;
        const int numTiles = ceilDiv(layer.convW, 32) * ceilDiv(layer.convH, 8) * ceilDiv(numBlocks, 2);
        if (numBlocks % 2 != 0 || numTiles < 1200)
          nf = 1;
        else if (numInputChannels > 64)
          mx = 3;

        layer.tileW = 16 * mx;
        layer.tileH = 8;
        layer.numFrags = nf;

        const std::string name = "conv_coopmat_mx" + std::to_string(mx) + "_nf" + std::to_string(nf) +
                                 (accF16 ? "_f16acc" : "");
        const uint32_t workGroupSize = 4 * ctx.getSubgroupSize();
        if ((result = ctx.getPipeline(name, {workGroupSize, upsample, pool, relu}, true, layer.pipeline)) != Result::Success)
          return result;
      }
      else
      {
        layer.tileW = 16;
        layer.tileH = 8;
        layer.numFrags = 1;
        if ((result = ctx.getPipeline("conv_simple", {128, upsample, pool, relu}, false, layer.pipeline)) != Result::Success)
          return result;
      }
    }

    return Result::Success;
  }

  Result DenoiserImpl::init(VkCommandBuffer cmd, const DenoiserCreateInfo& createInfo)
  {
    info = createInfo;
    if (!cmd || info.width == 0 || info.height == 0 || info.width > 65536 || info.height > 65536 ||
        (info.normal && !info.albedo) || (info.hdr && info.srgb))
      return Result::ErrorInvalidArgument;

    // Select the weights (same rules as Open Image Denoise)
    const void* blob = info.weights;
    size_t blobByteSize = info.weightsByteSize;
    if (blob)
      weightsName = "custom";
    else
    {
      std::string name = info.hdr ? "rt_hdr" : "rt_ldr";
      if (info.albedo && info.normal)
        name += info.cleanAux ? "_calb_cnrm" : "_alb_nrm";
      else if (info.albedo)
        name += "_alb";

      std::vector<std::string> candidates;
      if (info.quality == Quality::High)
        candidates = {name + "_large", name};
      else if (info.quality == Quality::Fast)
        candidates = {name + "_small", name};
      else
        candidates = {name};

      for (const auto& candidate : candidates)
      {
        if ((blob = getEmbeddedWeights(candidate.c_str(), &blobByteSize)) != nullptr)
        {
          weightsName = candidate;
          break;
        }
      }
      if (!blob)
        return Result::ErrorInvalidArgument; // model not embedded
    }

    weightsBlob.assign(static_cast<const uint8_t*>(blob), static_cast<const uint8_t*>(blob) + blobByteSize);
    if (!parseWeights(weightsBlob.data(), weightsBlob.size(), weights))
      return Result::ErrorInvalidArgument;
    const uint32_t expectedInputChannels = 3 * (1 + (info.albedo ? 1 : 0) + (info.normal ? 1 : 0));
    if (weights.header->inputChannels != expectedInputChannels)
      return Result::ErrorInvalidArgument;

    // Transfer function (same as Open Image Denoise)
    transfer = info.srgb ? Transfer::Linear : (info.hdr ? Transfer::PU : Transfer::SRGB);
    {
      const float yMax = 65504.f;
      const float xMax = (transfer == Transfer::PU) ? puForward(yMax) :
                         (transfer == Transfer::SRGB) ? srgbForward(yMax) : yMax;
      normScale = float(1. / xMax);
      rcpNormScale = xMax;
    }

    // Tiling (same as Open Image Denoise)
    H = int(info.height);
    W = int(info.width);
    const int tileAlignment = minTileAlignment;
    tileOverlap = roundUp(int(weights.header->receptiveField) / 2, tileAlignment);
    tileH = roundUp(H, minTileAlignment);
    tileW = roundUp(W, minTileAlignment);
    tilePadH = tileH % tileAlignment;
    tilePadW = tileW % tileAlignment;
    tileCountH = 1;
    tileCountW = 1;

    const int minTileDim = std::max(4 * tileOverlap, 768);
    const int minTileH = roundUp(minTileDim, tileAlignment, tilePadH);
    const int minTileW = roundUp(minTileDim, tileAlignment, tilePadW);
    const int maxTileSize = (info.maxMemoryMB == 0) ? defaultMaxTileSize : INT_MAX;
    const size_t maxMemoryByteSize = (info.maxMemoryMB == 0) ? SIZE_MAX : size_t(info.maxMemoryMB) * 1024 * 1024;

    for (;;)
    {
      scratchByteSize = planMemory();
      const bool fits = size_t(tileH) * size_t(tileW) <= size_t(maxTileSize) &&
                        scratchByteSize + weights.dataByteSize <= maxMemoryByteSize;
      if (fits)
        break;

      if (tileH > minTileH && tileH > tileW)
      {
        const int newTileH = ceilDiv(H + (2 * tileOverlap + tilePadH) * tileCountH, tileCountH + 1);
        tileH = std::clamp(roundUp(newTileH, tileAlignment, tilePadH), minTileH, tileH - tileAlignment);
        tileCountH = std::max(ceilDiv(H - (2 * tileOverlap + tilePadH), tileH - (2 * tileOverlap + tilePadH)), 1);
      }
      else if (tileW > minTileW)
      {
        const int newTileW = ceilDiv(W + (2 * tileOverlap + tilePadW) * tileCountW, tileCountW + 1);
        tileW = std::clamp(roundUp(newTileW, tileAlignment, tilePadW), minTileW, tileW - tileAlignment);
        tileCountW = std::max(ceilDiv(W - (2 * tileOverlap + tilePadW), tileW - (2 * tileOverlap + tilePadW)), 1);
      }
      else
        break; // cannot divide further
    }

    // Pipelines
    // FP16 accumulation for the faster quality modes (like Open Image Denoise)
    accF16 = ctx.isCooperativeMatrixEnabled() && ctx.hasCoopMatAccF16() && info.quality != Quality::High;
    Result result = createPipelines(accF16);
    if (result != Result::Success)
      return result;

    // Autoexposure (on the whole image)
    if (info.hdr)
    {
      aeNumBinsH = ceilDiv(H, autoexposureMaxBinSize);
      aeNumBinsW = ceilDiv(W, autoexposureMaxBinSize);
      const int numBins = aeNumBinsH * aeNumBinsW;
      aeNumGroups = std::min(ceilDiv(numBins, autoexposureGroupSize), autoexposureGroupSize);
      aeByteSize = roundUp(size_t(numBins) * 4 + size_t(aeNumGroups) * 8, memoryAlignment) + memoryAlignment;
      if ((result = ctx.createBuffer(aeByteSize, false, aeBuffer)) != Result::Success)
        return result;
    }

    // Memory
    if ((result = ctx.createBuffer(scratchByteSize, false, scratchBuffer)) != Result::Success ||
        (result = ctx.createBuffer(weights.dataByteSize, false, weightBuffer)) != Result::Success ||
        (result = ctx.createBuffer(weights.dataByteSize, true, stagingBuffer)) != Result::Success)
      return result;

    // Repack the weights of the layers using the experimental coopmat2 kernel: [tap][K][N] f16
    // weights and [N] f32 bias
    {
      std::vector<uint8_t> cm2Data;
      for (Layer& layer : layers)
      {
        if (!layer.cm2)
          continue;
        const uint32_t K = layer.desc.inChannels0 + layer.desc.inChannels1;
        const uint32_t N = layer.desc.outChannels;
        const uint32_t NB = N / 16;
        const uint8_t* data = static_cast<const uint8_t*>(weights.data);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(data + layer.desc.weightOffset);
        const float* bias = reinterpret_cast<const float*>(data + layer.desc.biasF32Offset);

        layer.cm2WeightOffset = roundUp(cm2Data.size(), memoryAlignment);
        cm2Data.resize(layer.cm2WeightOffset + size_t(9) * K * N * 2);
        uint16_t* dstW = reinterpret_cast<uint16_t*>(cm2Data.data() + layer.cm2WeightOffset);
        for (uint32_t kc = 0; kc < K / 16; ++kc)
          for (uint32_t tap = 0; tap < 9; ++tap)
            for (uint32_t nb = 0; nb < NB; ++nb)
              for (uint32_t ic = 0; ic < 16; ++ic)
                for (uint32_t oc = 0; oc < 16; ++oc)
                  dstW[(size_t(tap) * K + kc * 16 + ic) * N + nb * 16 + oc] =
                    src[((size_t(kc) * 9 + tap) * NB + nb) * 256 + ic * 16 + oc];

        layer.cm2BiasOffset = roundUp(cm2Data.size(), memoryAlignment);
        cm2Data.resize(layer.cm2BiasOffset + size_t(N) * 4);
        float* dstB = reinterpret_cast<float*>(cm2Data.data() + layer.cm2BiasOffset);
        for (uint32_t n = 0; n < N; ++n)
          dstB[n] = bias[(n / 16) * 256 + (n % 16)]; // first row of the replicated bias
      }

      if (!cm2Data.empty())
      {
        if ((result = ctx.createBuffer(cm2Data.size(), false, cm2Buffer)) != Result::Success ||
            (result = ctx.createBuffer(cm2Data.size(), true, cm2Staging)) != Result::Success)
          return result;
        memcpy(cm2Staging.mapped, cm2Data.data(), cm2Data.size());
        VkBufferCopy cm2Region{0, 0, cm2Data.size()};
        ctx.getFuncs().vkCmdCopyBuffer(cmd, cm2Staging.buffer, cm2Buffer.buffer, 1, &cm2Region);
      }
    }

    // Record the upload of the weights
    memcpy(stagingBuffer.mapped, weights.data, weights.dataByteSize);
    const VulkanFuncs& fn = ctx.getFuncs();
    VkBufferCopy region{0, 0, weights.dataByteSize};
    fn.vkCmdCopyBuffer(cmd, stagingBuffer.buffer, weightBuffer.buffer, 1, &region);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Profiling
    if (getenv("VK_OIDN_PROFILE") && atoi(getenv("VK_OIDN_PROFILE")) > 0)
    {
      profileNames.push_back("input");
      for (const Layer& layer : layers)
        profileNames.push_back(layer.desc.name);
      profileNames.push_back("output");
      profileTimes.assign(profileNames.size(), 0.);
      numQueries = uint32_t(profileNames.size()) * 2;

      VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
      queryInfo.queryCount = numQueries;
      if (fn.vkCreateQueryPool(ctx.getDevice(), &queryInfo, ctx.getAllocator(), &queryPool) != VK_SUCCESS)
        queryPool = VK_NULL_HANDLE;
    }

    // The packed weights are no longer needed on the host
    weightsBlob.clear();
    weightsBlob.shrink_to_fit();
    weights = Weights();

    return Result::Success;
  }

  void DenoiserImpl::cmdProfile(VkCommandBuffer cmd, uint32_t query)
  {
    if (profiling)
      ctx.getFuncs().vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, query);
  }

  void DenoiserImpl::collectProfile()
  {
    if (!profilePending)
      return;
    std::vector<uint64_t> timestamps(numQueries);
    if (ctx.getFuncs().vkGetQueryPoolResults(ctx.getDevice(), queryPool, 0, numQueries,
                                             timestamps.size() * sizeof(uint64_t), timestamps.data(),
                                             sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
      return; // not available yet

    for (size_t i = 0; i < profileTimes.size(); ++i)
      profileTimes[i] += double(timestamps[i * 2 + 1] - timestamps[i * 2]) * ctx.getTimestampPeriod() * 1e-6;
    ++profileCount;
    profilePending = false;
  }

  void DenoiserImpl::cmdAutoexposure(VkCommandBuffer cmd, const ImageParams& color, uint64_t resultAddress,
                                     const VkImageView* views)
  {
    const int numBins = aeNumBinsH * aeNumBinsW;
    const uint64_t bins   = aeBuffer.address;
    const uint64_t sums   = bins + uint64_t(numBins) * 4;
    const uint64_t counts = sums + uint64_t(aeNumGroups) * 4;

    AutoexposureDownsampleParams downsample{};
    downsample.src = color.ptr;
    downsample.bins = bins;
    downsample.srcHStride = color.hByteStride;
    downsample.srcFmt = color.fmt;
    downsample.H = H;
    downsample.W = W;
    if (views)
    {
      ctx.cmdPushImageViews(cmd, views);
      ctx.cmdDispatch(cmd, aeDownsampleImagePipeline, downsample, uint32_t(aeNumBinsW), uint32_t(aeNumBinsH), 1, true);
    }
    else
      ctx.cmdDispatch(cmd, aeDownsamplePipeline, downsample, uint32_t(aeNumBinsW), uint32_t(aeNumBinsH), 1);

    AutoexposureReduceParams reduce{};
    reduce.bins = bins;
    reduce.sums = sums;
    reduce.counts = counts;
    reduce.size = numBins;
    ctx.cmdDispatch(cmd, aeReducePipeline, reduce, uint32_t(aeNumGroups), 1, 1);

    AutoexposureReduceFinalParams reduceFinal{};
    reduceFinal.sums = sums;
    reduceFinal.counts = counts;
    reduceFinal.result = resultAddress;
    reduceFinal.size = aeNumGroups;
    ctx.cmdDispatch(cmd, aeReduceFinalPipeline, reduceFinal, 1, 1, 1);
  }

  Result DenoiserImpl::cmdDenoise(VkCommandBuffer cmd, const DenoiseInputs& inputs)
  {
    if (!cmd)
      return Result::ErrorInvalidArgument;

    // Either image views or buffers must be used for all images
    const bool useViews = inputs.colorView.view != VK_NULL_HANDLE;
    const VkImageView* views = nullptr;
    VkImageView viewArray[4] = {};
    ImageParams color, albedo, normal, output;

    if (useViews)
    {
      if (!ctx.areImageViewsSupported())
        return Result::ErrorUnsupportedDevice;
      if (!inputs.outputView.view ||
          info.albedo != (inputs.albedoView.view != VK_NULL_HANDLE) ||
          info.normal != (inputs.normalView.view != VK_NULL_HANDLE) ||
          inputs.color.address || inputs.albedo.address || inputs.normal.address || inputs.output.address)
        return Result::ErrorInvalidArgument;

      // Tiled denoising cannot be done in place
      if (getTileCount() > 1 && inputs.outputView.view == inputs.colorView.view)
        return Result::ErrorInvalidArgument;

      // Unused bindings are set to the color view to keep all descriptors valid
      viewArray[0] = inputs.colorView.view;
      viewArray[1] = info.albedo ? inputs.albedoView.view : inputs.colorView.view;
      viewArray[2] = info.normal ? inputs.normalView.view : inputs.colorView.view;
      viewArray[3] = inputs.outputView.view;
      views = viewArray;
      output.fmt = 3u << 24; // 3 channels
    }
    else
    {
      if (!isValidImage(inputs.color) || !isValidImage(inputs.output) ||
          info.albedo != (inputs.albedo.address != 0) || info.normal != (inputs.normal.address != 0) ||
          (info.albedo && !isValidImage(inputs.albedo)) || (info.normal && !isValidImage(inputs.normal)) ||
          inputs.albedoView.view || inputs.normalView.view || inputs.outputView.view)
        return Result::ErrorInvalidArgument;

      // Tiled denoising cannot be done in place
      if (getTileCount() > 1 && inputs.output.address == inputs.color.address)
        return Result::ErrorInvalidArgument;

      color  = getImageParams(inputs.color, info.width);
      albedo = info.albedo ? getImageParams(inputs.albedo, info.width) : ImageParams();
      normal = info.normal ? getImageParams(inputs.normal, info.width) : ImageParams();
      output = getImageParams(inputs.output, info.width);
    }

    // Previous executions may still access the internal resources
    ctx.cmdComputeBarrier(cmd);

    if (queryPool)
    {
      // Profile one execution at a time: collect the results of the previous one if available
      collectProfile();
      if (!profilePending)
      {
        ctx.getFuncs().vkCmdResetQueryPool(cmd, queryPool, 0, numQueries);
        profiling = true;
        profilePending = true;
      }
    }

    // Input scale
    uint64_t inputScalePtr = 0;
    float inputScale = 1.f;
    if (std::isnan(inputs.inputScale))
    {
      if (info.hdr)
      {
        inputScalePtr = aeBuffer.address + aeByteSize - memoryAlignment;
        cmdAutoexposure(cmd, color, inputScalePtr, views);
      }
    }
    else
      inputScale = inputs.inputScale;
    const float outputScale = (inputScale != 0.f) ? 1.f / inputScale : 0.f;

    const uint32_t flags = uint32_t(transfer) | (info.hdr ? 4u : 0u);
    // Presence of the auxiliary images for the image view kernels
    const uint32_t inputFlags = flags | (useViews && info.albedo ? 16u : 0u) | (useViews && info.normal ? 32u : 0u);
    const uint64_t scratch = scratchBuffer.address;
    const Tensor& inputTensor = tensors.front();
    const Tensor& outputTensor = tensors.back();

    // Iterate over the tiles (same as Open Image Denoise)
    for (int i = 0; i < tileCountH; ++i)
    {
      const int h = i * (tileH - (2 * tileOverlap + tilePadH)); // input tile position (including overlaps)
      const int overlapBeginH = i > 0 ? tileOverlap : 0;
      const int overlapEndH = i < tileCountH - 1 ? tileOverlap + tilePadH : 0;
      const int tileH1 = std::min(H - h, tileH); // input tile size (including overlaps)
      const int tileH2 = tileH1 - overlapBeginH - overlapEndH; // output tile size
      const int alignOffsetH = tileH - roundUp(tileH1, minTileAlignment); // align to the bottom

      for (int j = 0; j < tileCountW; ++j)
      {
        const int w = j * (tileW - (2 * tileOverlap + tilePadW));
        const int overlapBeginW = j > 0 ? tileOverlap : 0;
        const int overlapEndW = j < tileCountW - 1 ? tileOverlap + tilePadW : 0;
        const int tileW1 = std::min(W - w, tileW);
        const int tileW2 = tileW1 - overlapBeginW - overlapEndW;
        const int alignOffsetW = tileW - roundUp(tileW1, minTileAlignment);

        // Input processing
        InputProcessParams in{};
        in.dst = scratch + inputTensor.offset;
        in.inputScalePtr = inputScalePtr;
        in.inputPtr = color.ptr;
        in.albedoPtr = albedo.ptr;
        in.normalPtr = normal.ptr;
        in.inputHStride = color.hByteStride;
        in.inputFmt = color.fmt;
        in.albedoHStride = albedo.hByteStride;
        in.albedoFmt = albedo.fmt;
        in.normalHStride = normal.hByteStride;
        in.normalFmt = normal.fmt;
        in.dstH = inputTensor.H;
        in.dstW = inputTensor.W;
        in.dstC = inputTensor.C;
        in.flags = inputFlags;
        in.hSrcBegin = h;
        in.wSrcBegin = w;
        in.hDstBegin = alignOffsetH;
        in.wDstBegin = alignOffsetW;
        in.tileH = tileH1;
        in.tileW = tileW1;
        in.inputScale = inputScale;
        in.normScale = normScale;
        cmdProfile(cmd, 0);
        if (useViews)
          ctx.cmdPushImageViews(cmd, views);
        ctx.cmdDispatch(cmd, useViews ? inputImagePipeline : inputPipeline, in,
                        ceilDiv(uint32_t(inputTensor.W), elementwiseGroupSizeX),
                        ceilDiv(uint32_t(inputTensor.H), elementwiseGroupSizeY), 1, useViews);
        cmdProfile(cmd, 1);

        // Convolutions
        for (size_t l = 0; l < layers.size(); ++l)
        {
          const Layer& layer = layers[l];
          cmdProfile(cmd, uint32_t(l + 1) * 2);
          const Tensor& src0 = tensors[layer.src0];
          ConvParams conv{};
          conv.src0 = scratch + src0.offset;
          conv.src1 = (layer.src1 >= 0) ? scratch + tensors[layer.src1].offset : 0;
          conv.weight = weightBuffer.address + layer.desc.weightOffset;
          conv.bias = weightBuffer.address + (accF16 ? layer.desc.biasF16Offset : layer.desc.biasF32Offset);
          if (layer.cm2)
          {
            conv.weight = cm2Buffer.address + layer.cm2WeightOffset;
            conv.bias = cm2Buffer.address + layer.cm2BiasOffset;
          }
          conv.dst = scratch + tensors[layer.dst].offset;
          conv.H = layer.convH;
          conv.W = layer.convW;
          conv.C0 = int(layer.desc.inChannels0);
          conv.C1 = int(layer.desc.inChannels1);
          conv.NB = int(layer.desc.outChannels / 16);
          ctx.cmdDispatch(cmd, layer.pipeline, conv,
                          ceilDiv(uint32_t(layer.convW), uint32_t(layer.tileW)),
                          ceilDiv(uint32_t(layer.convH), uint32_t(layer.tileH)),
                          ceilDiv(uint32_t(conv.NB), uint32_t(layer.numFrags)));
          cmdProfile(cmd, uint32_t(l + 1) * 2 + 1);
        }

        // Output processing
        OutputProcessParams out{};
        out.src = scratch + outputTensor.offset;
        out.dst = output.ptr;
        out.inputScalePtr = inputScalePtr;
        out.dstHStride = output.hByteStride;
        out.dstFmt = output.fmt;
        out.srcW = outputTensor.W;
        out.srcC = outputTensor.C;
        out.hSrcBegin = alignOffsetH + overlapBeginH;
        out.wSrcBegin = alignOffsetW + overlapBeginW;
        out.hDstBegin = h + overlapBeginH;
        out.wDstBegin = w + overlapBeginW;
        out.tileH = tileH2;
        out.tileW = tileW2;
        out.flags = flags | (useViews && isHalfFormat(inputs.outputView.format) ? 64u : 0u);
        out.outputScale = outputScale;
        out.rcpNormScale = rcpNormScale;
        cmdProfile(cmd, numQueries - 2);
        if (useViews)
          ctx.cmdPushImageViews(cmd, views);
        ctx.cmdDispatch(cmd, useViews ? outputImagePipeline : outputPipeline, out,
                        ceilDiv(uint32_t(tileW2), elementwiseGroupSizeX),
                        ceilDiv(uint32_t(tileH2), elementwiseGroupSizeY), 1, useViews);
        cmdProfile(cmd, numQueries - 1);
        profiling = false; // only the first tile is profiled
      }
    }

    return Result::Success;
  }

} // namespace vkoidn
