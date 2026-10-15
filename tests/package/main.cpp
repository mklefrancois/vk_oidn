// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Links against the installed vk_oidn SDK package. Referencing Context::create and Denoiser::create
// pulls the kernels and the embedded weights into the link, so missing symbols are detected.

#include <vk_oidn/vk_oidn.h>
#include <cstdio>

int main()
{
  volatile auto createContext = &vkoidn::Context::create;
  volatile auto createDenoiser = &vkoidn::Denoiser::create;
  volatile auto getExtensions = &vkoidn::getRequiredDeviceExtensions;
  const bool linked = createContext && createDenoiser && getExtensions;

  printf("vk_oidn %d.%d.%d: %s\n", VK_OIDN_VERSION_MAJOR, VK_OIDN_VERSION_MINOR, VK_OIDN_VERSION_PATCH,
         vkoidn::getResultString(vkoidn::Result::Success));
  return linked ? 0 : 1;
}
