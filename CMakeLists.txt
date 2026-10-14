# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)
project(vk_oidn VERSION 0.1.0 LANGUAGES CXX)

option(VK_OIDN_EMBED_WEIGHTS "Embed the packed weights from the weights folder in the library" ON)
option(VK_OIDN_BUILD_TESTS "Build the vk_oidn tests" ${PROJECT_IS_TOP_LEVEL})

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
  src/weights.cpp
  src/weights.h
  src/weights_format.h
)
add_library(vk_oidn::vk_oidn ALIAS vk_oidn)

target_include_directories(vk_oidn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(vk_oidn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vk_oidn PUBLIC Vulkan::Headers)
target_compile_definitions(vk_oidn PRIVATE VK_NO_PROTOTYPES) # functions are loaded at runtime

# Embedded weights
file(GLOB VK_OIDN_WEIGHT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/weights/*.vkoidnw)
if(VK_OIDN_EMBED_WEIGHTS)
  find_package(Python3 COMPONENTS Interpreter)
  if(NOT Python3_FOUND)
    message(WARNING "vk_oidn: Python 3 not found, the weights will not be embedded")
    set(VK_OIDN_EMBED_WEIGHTS OFF)
  endif()
endif()

if(VK_OIDN_EMBED_WEIGHTS)
  set(VK_OIDN_EMBEDDED_WEIGHTS_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/weights_embedded.cpp)
  add_custom_command(
    OUTPUT ${VK_OIDN_EMBEDDED_WEIGHTS_CPP}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_files.py
            -o ${VK_OIDN_EMBEDDED_WEIGHTS_CPP} -n getEmbeddedWeightsFile ${VK_OIDN_WEIGHT_FILES}
    DEPENDS ${VK_OIDN_WEIGHT_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_files.py
    COMMENT "vk_oidn: embedding weights"
    VERBATIM
  )
  target_sources(vk_oidn PRIVATE ${VK_OIDN_EMBEDDED_WEIGHTS_CPP})
  target_compile_definitions(vk_oidn PRIVATE VK_OIDN_EMBEDDED_WEIGHTS)
  list(LENGTH VK_OIDN_WEIGHT_FILES VK_OIDN_NUM_WEIGHT_FILES)
  message(STATUS "vk_oidn: embedding ${VK_OIDN_NUM_WEIGHT_FILES} weight files")
endif()

# Tests
if(VK_OIDN_BUILD_TESTS)
  enable_testing()

  add_executable(vk_oidn_weights_test tests/weights_test.cpp)
  target_link_libraries(vk_oidn_weights_test PRIVATE vk_oidn)
  target_include_directories(vk_oidn_weights_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME weights COMMAND vk_oidn_weights_test ${VK_OIDN_WEIGHT_FILES})
endif()
