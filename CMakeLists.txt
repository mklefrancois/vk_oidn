# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

# MSVC debug information format selected with MSVC_DEBUG_INFORMATION_FORMAT (CMake 3.25+)
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

project(vk_oidn VERSION 0.2.0 LANGUAGES CXX)

option(VK_OIDN_EMBED_WEIGHTS "Embed the packed weights from the weights folder in the library" ON)
option(VK_OIDN_BUILD_TESTS "Build the vk_oidn tests" ${PROJECT_IS_TOP_LEVEL})
option(VK_OIDN_INSTALL "Generate the install rules (SDK package)" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan headers (the loader is not linked: the application provides vkGet*ProcAddr)
find_package(Vulkan REQUIRED OPTIONAL_COMPONENTS glslangValidator)

# Slang compiler for the shaders (from the Vulkan SDK by default)
find_program(VK_OIDN_SLANGC slangc
  HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin
  DOC "Path to the Slang compiler")
if(NOT VK_OIDN_SLANGC)
  message(FATAL_ERROR "slangc not found, please install the Vulkan SDK or set VK_OIDN_SLANGC")
endif()

# The kernels need a recent Slang compiler (-denorm-mode-fp16). Older versions, such as the slangc
# of some Linux distribution packages, do not have it. The version string is not reliable across
# packages, so the option itself is checked.
execute_process(COMMAND ${VK_OIDN_SLANGC} -h OUTPUT_VARIABLE slangc_help ERROR_VARIABLE slangc_help)
if(NOT slangc_help MATCHES "denorm-mode-fp16")
  message(FATAL_ERROR "vk_oidn: ${VK_OIDN_SLANGC} is too old (no -denorm-mode-fp16 option). "
                      "Install a recent Vulkan SDK (tested with 1.4.363, Slang 2026.17) and set VULKAN_SDK, "
                      "or set VK_OIDN_SLANGC to a recent slangc (https://github.com/shader-slang/slang/releases)")
endif()
execute_process(COMMAND ${VK_OIDN_SLANGC} -version OUTPUT_VARIABLE slangc_version ERROR_VARIABLE slangc_version
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
message(STATUS "vk_oidn: using Slang compiler ${VK_OIDN_SLANGC} (${slangc_version})")

add_library(vk_oidn STATIC
  include/vk_oidn/vk_oidn.h
  src/context.cpp
  src/context.h
  src/denoiser.cpp
  src/kernel_params.h
  src/vk_oidn.cpp
  src/vulkan_funcs.h
  src/weights.cpp
  src/weights.h
  src/weights_format.h
)
add_library(vk_oidn::vk_oidn ALIAS vk_oidn)

target_include_directories(vk_oidn PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(vk_oidn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vk_oidn PUBLIC Vulkan::Headers)
target_compile_definitions(vk_oidn PRIVATE VK_NO_PROTOTYPES) # functions are loaded at runtime
# Debug library named vk_oidnd so both configurations can be installed side by side; PIC so the
# static library can be linked into shared libraries
set_target_properties(vk_oidn PROPERTIES DEBUG_POSTFIX d POSITION_INDEPENDENT_CODE ON)
# MSVC: debug information embedded in the objects of the static library (/Z7) instead of a separate
# PDB, so the Debug library of the SDK package is self-contained (no LNK4099 warning, and the
# debugger can step into vk_oidn)
set_target_properties(vk_oidn PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")

# The version in the header must match the project version
file(STRINGS include/vk_oidn/vk_oidn.h version_lines REGEX "#define VK_OIDN_VERSION_(MAJOR|MINOR|PATCH) ")
string(REGEX REPLACE "[^;]*MAJOR ([0-9]+)[^;]*;[^;]*MINOR ([0-9]+)[^;]*;[^;]*PATCH ([0-9]+).*" "\\1.\\2.\\3" header_version "${version_lines}")
if(NOT header_version VERSION_EQUAL PROJECT_VERSION)
  message(FATAL_ERROR "vk_oidn: version ${header_version} in vk_oidn.h does not match the project version ${PROJECT_VERSION}")
endif()
if(MSVC)
  target_compile_definitions(vk_oidn PRIVATE _CRT_SECURE_NO_WARNINGS) # getenv() for the debug variables
endif()

# Kernels: <name>|<entry point>|<comma-separated defines>
# Each kernel is compiled with float controls (FP16 round-to-nearest-even and denormal
# preservation) and without them (suffix "_nofc") for devices which do not support them.
set(VK_OIDN_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vk_oidn.slang)
set(VK_OIDN_KERNELS
  "input_process|inputProcess|"
  "input_process_image|inputProcess|IMAGE_IO"
  "output_process|outputProcess|"
  "output_process_image|outputProcess|IMAGE_IO"
  "image_copy|imageCopy|"
  "autoexposure_downsample|autoexposureDownsample|"
  "autoexposure_downsample_image|autoexposureDownsample|IMAGE_IO"
  "autoexposure_reduce|autoexposureReduce|"
  "autoexposure_reduce_final|autoexposureReduceFinal|"
  "conv_simple|convSimple|"
  "conv_coopmat_mx2_nf2|convCoopMat|MX=2,N_FRAGS=2"
  "conv_coopmat_mx2_nf1|convCoopMat|MX=2,N_FRAGS=1"
  "conv_coopmat_mx2_nf2_f16acc|convCoopMat|MX=2,N_FRAGS=2,ACC_F16"
  "conv_coopmat_mx2_nf1_f16acc|convCoopMat|MX=2,N_FRAGS=1,ACC_F16"
  "conv_coopmat_up_nf2|convCoopMatUp|MX=2,N_FRAGS=2"
  "conv_coopmat_up_nf1|convCoopMatUp|MX=2,N_FRAGS=1"
  "conv_coopmat_up_nf2_f16acc|convCoopMatUp|MX=2,N_FRAGS=2,ACC_F16"
  "conv_coopmat_up_nf1_f16acc|convCoopMatUp|MX=2,N_FRAGS=1,ACC_F16"
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(VK_OIDN_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
set(VK_OIDN_SPV_FILES)
foreach(kernel ${VK_OIDN_KERNELS})
  string(REPLACE "|" ";" kernel_parts "${kernel}")
  list(GET kernel_parts 0 kernel_name)
  list(GET kernel_parts 1 kernel_entry)
  list(GET kernel_parts 2 kernel_defines)

  set(define_args)
  if(kernel_defines)
    string(REPLACE "," ";" kernel_defines "${kernel_defines}")
    foreach(def ${kernel_defines})
      list(APPEND define_args "-D${def}")
    endforeach()
  endif()

  set(spv ${VK_OIDN_SPV_DIR}/${kernel_name}.spv)
  set(spv_nofc ${VK_OIDN_SPV_DIR}/${kernel_name}_nofc.spv)
  set(spv_tmp ${VK_OIDN_SPV_DIR}/${kernel_name}_tmp.spv)
  set(slangc_args -target spirv -profile spirv_1_5 -O1 -warnings-disable 41012 -default-image-format-unknown -entry ${kernel_entry} ${define_args})
  add_custom_command(
    OUTPUT ${spv} ${spv_nofc}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VK_OIDN_SPV_DIR}
    COMMAND ${VK_OIDN_SLANGC} ${VK_OIDN_SHADER} ${slangc_args} -denorm-mode-fp16 preserve -o ${spv_tmp}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/spirv_float_controls.py ${spv_tmp} ${spv}
    COMMAND ${CMAKE_COMMAND} -E remove ${spv_tmp}
    COMMAND ${VK_OIDN_SLANGC} ${VK_OIDN_SHADER} ${slangc_args} -o ${spv_nofc}
    DEPENDS ${VK_OIDN_SHADER} ${CMAKE_CURRENT_SOURCE_DIR}/tools/spirv_float_controls.py
    COMMENT "vk_oidn: compiling kernel ${kernel_name}"
    VERBATIM
  )
  list(APPEND VK_OIDN_SPV_FILES ${spv} ${spv_nofc})
endforeach()

# Experimental kernels written in GLSL (VK_NV_cooperative_matrix2), compiled with glslang
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
  set(cm2_src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/experimental/conv_cm2.comp)
  foreach(variant "conv_cm2|" "conv_cm2_nofc|-DOIDN_NO_FLOAT_CONTROLS")
    string(REPLACE "|" ";" variant_parts "${variant}")
    list(GET variant_parts 0 kernel_name)
    list(LENGTH variant_parts num_parts)
    set(define_args)
    if(num_parts GREATER 1)
      list(GET variant_parts 1 define_args)
    endif()
    set(spv ${VK_OIDN_SPV_DIR}/${kernel_name}.spv)
    add_custom_command(
      OUTPUT ${spv}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${VK_OIDN_SPV_DIR}
      COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 --quiet ${define_args} -o ${spv} ${cm2_src}
      DEPENDS ${cm2_src}
      COMMENT "vk_oidn: compiling experimental kernel ${kernel_name}"
      VERBATIM
    )
    list(APPEND VK_OIDN_SPV_FILES ${spv})
  endforeach()
  target_sources(vk_oidn PRIVATE ${cm2_src})
endif()

set(VK_OIDN_KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/kernels_embedded.cpp)
add_custom_command(
  OUTPUT ${VK_OIDN_KERNELS_CPP}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_files.py
          -o ${VK_OIDN_KERNELS_CPP} -n getEmbeddedKernel ${VK_OIDN_SPV_FILES}
  DEPENDS ${VK_OIDN_SPV_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_files.py
  COMMENT "vk_oidn: embedding kernels"
  VERBATIM
)
target_sources(vk_oidn PRIVATE ${VK_OIDN_KERNELS_CPP} src/kernels.h ${VK_OIDN_SHADER})
source_group("Shaders" FILES ${VK_OIDN_SHADER})

# Embedded weights
file(GLOB VK_OIDN_WEIGHT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/weights/*.vkoidnw)
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

  add_executable(vk_oidn_denoise_test tests/denoise_test.cpp)
  target_link_libraries(vk_oidn_denoise_test PRIVATE vk_oidn Vulkan::Vulkan)
endif()

# Install rules: the SDK package (headers, static libraries with the embedded kernels and weights,
# CMake package configuration). Consumers use find_package(vk_oidn CONFIG) and vk_oidn::vk_oidn.
if(VK_OIDN_INSTALL)
  include(CMakePackageConfigHelpers)
  set(config_dir lib/cmake/vk_oidn)
  install(TARGETS vk_oidn EXPORT vk_oidnTargets ARCHIVE DESTINATION lib)
  install(FILES include/vk_oidn/vk_oidn.h DESTINATION include/vk_oidn)
  install(EXPORT vk_oidnTargets NAMESPACE vk_oidn:: DESTINATION ${config_dir})
  configure_package_config_file(cmake/vk_oidnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vk_oidnConfig.cmake INSTALL_DESTINATION ${config_dir})
  # 0.x versions may change the API in minor versions
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/vk_oidnConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vk_oidnConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/vk_oidnConfigVersion.cmake DESTINATION ${config_dir})
  install(FILES LICENSE NOTICE README.md DESTINATION .)
endif()
