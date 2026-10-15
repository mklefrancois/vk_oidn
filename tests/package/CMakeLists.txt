# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

# Checks that an installed vk_oidn SDK package can be found and linked, like an application would.
#   cmake -S tests/package -B build_package_test -Dvk_oidn_DIR=<sdk>/lib/cmake/vk_oidn
#   cmake --build build_package_test --config Release
#   build_package_test/[Release/]vk_oidn_package_test

cmake_minimum_required(VERSION 3.21)
project(vk_oidn_package_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vk_oidn 0.1 CONFIG REQUIRED)
message(STATUS "Found vk_oidn ${vk_oidn_VERSION} in ${vk_oidn_DIR}")

add_executable(vk_oidn_package_test main.cpp)
target_link_libraries(vk_oidn_package_test PRIVATE vk_oidn::vk_oidn)
