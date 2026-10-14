// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace vkoidn {

  // Returns the embedded SPIR-V of the kernel with the specified name (nullptr if not found).
  // Kernel names: input_process, output_process, image_copy, autoexposure_downsample,
  // autoexposure_reduce, autoexposure_reduce_final, conv_simple,
  // conv_coopmat_mx<2|3>_nf<1|2>[_f16acc], each also with the suffix "_nofc" for the variant
  // without float controls (FP16 round-to-nearest-even and denormal preservation).
  const void* getEmbeddedKernel(const char* name, size_t* byteSize);

} // namespace vkoidn
