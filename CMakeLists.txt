# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)
project(vk_oidn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan headers (the loader is not linked: the application provides vkGet*ProcAddr)
find_package(Vulkan REQUIRED)

# Slang compiler for the shaders (from the Vulkan SDK by default)
find_program(VK_OIDN_SLANGC slangc
  HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin
  DOC "Path to the Slang compiler")
if(NOT VK_OIDN_SLANGC)
  message(FATAL_ERROR "slangc not found, please install the Vulkan SDK or set VK_OIDN_SLANGC")
endif()
message(STATUS "vk_oidn: using Slang compiler ${VK_OIDN_SLANGC}")

add_library(vk_oidn STATIC
  include/vk_oidn/vk_oidn.h
  src/vk_oidn.cpp
)
add_library(vk_oidn::vk_oidn ALIAS vk_oidn)

target_include_directories(vk_oidn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vk_oidn PUBLIC Vulkan::Headers)
target_compile_definitions(vk_oidn PRIVATE VK_NO_PROTOTYPES) # functions are loaded at runtime
