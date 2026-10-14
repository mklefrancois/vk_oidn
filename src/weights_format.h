// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Packed network weights format (produced by tools/pack_weights.py)
//
// All values are little-endian. The file consists of:
//   - WeightsHeader
//   - WeightsLayer[numLayers]
//   - data section at dataOffset (256-byte aligned), which is uploaded to the GPU as is
//
// For each layer, the data section contains (each part 256-byte aligned):
//   - weights: FP16, [K chunk][tap][output block][16 input channels][16 output channels]
//     where K chunk = 16 input channels (first source channels, then second source channels),
//     tap = ky * 3 + kx, output block = 16 output channels
//   - bias (FP32 accumulation): FP32, [output block][16 rows][16 output channels]
//     (the bias is replicated in 16 rows to be loaded as an accumulator matrix)
//   - bias (FP16 accumulation): FP16, same layout
//
// Channels are zero-padded to multiples of 16. Layers are in execution order.

#pragma once

#include <cstdint>

namespace vkoidn {

  constexpr char     weightsMagic[8] = {'V', 'K', 'O', 'I', 'D', 'N', 'W', '1'};
  constexpr uint32_t weightsVersion  = 1;

  constexpr int32_t weightsSrcInput = -1; // the source is the network input
  constexpr int32_t weightsSrcNone  = -2; // no source

  enum WeightsLayerFlags : uint32_t
  {
    weightsLayerReLU       = 1u << 0, // ReLU activation
    weightsLayerPoolDst    = 1u << 1, // 2x2 max pooling of the output
    weightsLayerUpsampleSrc0 = 1u << 2, // 2x nearest upsampling of the first source
  };

  enum WeightsFlags : uint32_t
  {
    weightsLargeNetwork = 1u << 0,
  };

  struct WeightsHeader
  {
    char     magic[8];
    uint32_t version;
    uint32_t headerByteSize;  // sizeof(WeightsHeader)
    uint32_t numLayers;
    uint32_t inputChannels;   // logical number of input channels (3, 6 or 9)
    uint32_t outputChannels;  // logical number of output channels (3)
    uint32_t receptiveField;  // in pixels, used to compute the tile overlap
    uint32_t flags;           // WeightsFlags
    uint32_t reserved[3];
    uint64_t dataOffset;      // offset of the data section in the file
    uint64_t dataByteSize;    // size of the data section
  };

  struct WeightsLayer
  {
    char     name[24];          // null-terminated
    int32_t  src0;              // index of the first source layer or weightsSrcInput
    int32_t  src1;              // index of the second (concatenated) source layer, weightsSrcInput or weightsSrcNone
    uint32_t flags;             // WeightsLayerFlags
    uint32_t inChannels0;       // padded channels of the first source
    uint32_t inChannels1;       // padded channels of the second source (0 if none)
    uint32_t outChannels;       // padded output channels
    uint32_t outChannelsLogical;
    uint32_t weightOffset;      // offset of the weights in the data section
    uint32_t biasF32Offset;     // offset of the FP32 bias in the data section
    uint32_t biasF16Offset;     // offset of the FP16 bias in the data section
  };

  static_assert(sizeof(WeightsHeader) == 64, "unexpected WeightsHeader size");
  static_assert(sizeof(WeightsLayer) == 64, "unexpected WeightsLayer size");

} // namespace vkoidn
