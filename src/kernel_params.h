// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Kernel parameters passed as push constants (must match the entry point parameters in
// shaders/vk_oidn.slang, std430 layout)

#pragma once

#include <cstdint>

namespace vkoidn {

  constexpr uint32_t maxParamsByteSize = 128;

  // Packed image format (see shaders/vk_oidn.slang)
  struct ImageParams
  {
    uint64_t ptr = 0;
    uint32_t hByteStride = 0;
    uint32_t fmt = 0; // bits 0-23: pixel byte stride, bits 24-25: channels, bit 28: half
  };

  struct InputProcessParams
  {
    uint64_t dst;
    uint64_t inputScalePtr;
    uint64_t inputPtr;
    uint64_t albedoPtr;
    uint64_t normalPtr;
    uint32_t inputHStride;
    uint32_t inputFmt;
    uint32_t albedoHStride;
    uint32_t albedoFmt;
    uint32_t normalHStride;
    uint32_t normalFmt;
    int32_t dstH;
    int32_t dstW;
    int32_t dstC;
    uint32_t flags; // bits 0-1: transfer function, bit 2: hdr, bit 3: snorm
    int32_t hSrcBegin;
    int32_t wSrcBegin;
    int32_t hDstBegin;
    int32_t wDstBegin;
    int32_t tileH;
    int32_t tileW;
    float inputScale;
    float normScale;
  };

  struct OutputProcessParams
  {
    uint64_t src;
    uint64_t dst;
    uint64_t inputScalePtr;
    uint32_t dstHStride;
    uint32_t dstFmt;
    int32_t srcW;
    int32_t srcC;
    int32_t hSrcBegin;
    int32_t wSrcBegin;
    int32_t hDstBegin;
    int32_t wDstBegin;
    int32_t tileH;
    int32_t tileW;
    uint32_t flags;
    float outputScale;
    float rcpNormScale;
  };

  struct AutoexposureDownsampleParams
  {
    uint64_t src;
    uint64_t bins;
    uint32_t srcHStride;
    uint32_t srcFmt;
    int32_t H;
    int32_t W;
  };

  struct AutoexposureReduceParams
  {
    uint64_t bins;
    uint64_t sums;
    uint64_t counts;
    int32_t size;
  };

  struct AutoexposureReduceFinalParams
  {
    uint64_t sums;
    uint64_t counts;
    uint64_t result;
    int32_t size;
  };

  struct ConvParams
  {
    uint64_t src0;
    uint64_t src1;
    uint64_t weight;
    uint64_t bias;
    uint64_t dst;
    int32_t H;
    int32_t W;
    int32_t C0;
    int32_t C1;
    int32_t NB;
  };

  static_assert(sizeof(InputProcessParams) <= maxParamsByteSize, "parameters too large");
  static_assert(sizeof(OutputProcessParams) <= maxParamsByteSize, "parameters too large");

  // Autoexposure constants (must match the shaders)
  constexpr int autoexposureMaxBinSize = 16;
  constexpr int autoexposureGroupSize = 256;

  // Elementwise kernel workgroup size (must match the shaders)
  constexpr uint32_t elementwiseGroupSizeX = 64;
  constexpr uint32_t elementwiseGroupSizeY = 4;

} // namespace vkoidn
