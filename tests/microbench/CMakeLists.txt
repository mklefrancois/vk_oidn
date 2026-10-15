# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

# Tensor core micro-benchmark harness (see README.md); not part of the vk_oidn build
cmake_minimum_required(VERSION 3.21)
project(vk_oidn_microbench CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Vulkan REQUIRED)
add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE Vulkan::Vulkan)
