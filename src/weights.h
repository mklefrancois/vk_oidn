// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "weights_format.h"
#include <cstddef>
#include <string>
#include <vector>

namespace vkoidn {

  // Validated view of packed weights (does not own the memory)
  struct Weights
  {
    const WeightsHeader* header = nullptr;
    const WeightsLayer* layers = nullptr; // header->numLayers layers
    const void* data = nullptr;           // data section to upload to the GPU
    size_t dataByteSize = 0;
  };

  // Parses and validates packed weights. Returns false and sets the error message if invalid.
  bool parseWeights(const void* blob, size_t byteSize, Weights& weights, std::string* error = nullptr);

  // Returns the embedded packed weights with the specified name (e.g. "rt_hdr_alb_nrm"), or nullptr
  // if not embedded
  const void* getEmbeddedWeights(const char* name, size_t* byteSize);

} // namespace vkoidn
