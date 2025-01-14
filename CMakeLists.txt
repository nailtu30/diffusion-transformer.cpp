cmake_minimum_required(VERSION 3.12)
project("diffusion-transformer")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(DIT_STANDALONE ON)
else()
    set(DIT_STANDALONE OFF)
endif()

#
# Option list
#

# general
#option(DIT_BUILD_TESTS                "DiT: build tests"    ${DIT_STANDALONE})
option(DIT_BUILD_EXAMPLES             "DiT: build examples" ${DIT_STANDALONE})
option(DIT_CUDA                       "DiT: cuda backend" OFF)
option(DIT_HIPBLAS                    "DiT: rocm backend" OFF)
option(DIT_METAL                      "DiT: metal backend" OFF)
option(DIT_VULKAN                     "DiT: vulkan backend" OFF)
option(DIT_SYCL                       "DiT: sycl backend" OFF)
option(DIT_MUSA                       "DiT: musa backend" OFF)
option(DIT_FAST_SOFTMAX               "DiT: x1.5 faster softmax, indeterministic (sometimes, same seed don't generate same image), cuda only" OFF)
option(DIT_BUILD_SHARED_LIBS          "DiT: build shared libs" OFF)
#option(DIT_BUILD_SERVER               "DiT: build server example"                           ON)

if(DIT_CUDA)
    message("-- Use CUDA as backend diffusion-transformer")
    set(GGML_CUDA ON)
    add_definitions(-DDIT_USE_CUDA)
    SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

if(DIT_METAL)
    message("-- Use Metal as backend diffusion-transformer")
    set(GGML_METAL ON)
    add_definitions(-DDIT_USE_METAL)
endif()

if (DIT_VULKAN)
    message("-- Use Vulkan as backend diffusion-transformer")
    set(GGML_VULKAN ON)
    add_definitions(-DDIT_USE_VULKAN)
endif ()

if (DIT_HIPBLAS)
    message("-- Use HIPBLAS as backend diffusion-transformer")
    set(GGML_HIPBLAS ON)
    add_definitions(-DDIT_USE_CUDA)
    if(DIT_FAST_SOFTMAX)
        set(GGML_CUDA_FAST_SOFTMAX ON)
    endif()
endif ()

if(DIT_MUSA)
    message("-- Use MUSA as backend diffusion-transformer")
    set(GGML_MUSA ON)
    add_definitions(-DDIT_USE_CUBLAS)
    if(DIT_FAST_SOFTMAX)
        set(GGML_CUDA_FAST_SOFTMAX ON)
    endif()
endif()

set(DIT_LIB diffusion-transformer)

file(GLOB DIT_LIB_SOURCES
    "*.h"
    "*.cpp"
    "*.hpp"
)

# we can get only one share lib
if(DIT_BUILD_SHARED_LIBS)
    message("-- Build shared library")
    message(${DIT_LIB_SOURCES})
    set(BUILD_SHARED_LIBS OFF)
    add_library(${DIT_LIB} SHARED ${DIT_LIB_SOURCES})
    add_definitions(-DDIT_BUILD_SHARED_LIB)
    target_compile_definitions(${DIT_LIB} PRIVATE -DDIT_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message("-- Build static library")
    set(BUILD_SHARED_LIBS OFF)
    add_library(${DIT_LIB} STATIC ${DIT_LIB_SOURCES})
endif()

if(DIT_SYCL)
    message("-- Use SYCL as backend diffusion-transformer")
    set(GGML_SYCL ON)
    add_definitions(-DDIT_USE_SYCL)
    # disable fast-math on host, see:
    # https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/fp-model-fp.html
    if (WIN32)
        set(SYCL_COMPILE_OPTIONS /fp:precise)
    else()
        set(SYCL_COMPILE_OPTIONS -fp-model=precise)
    endif()
    message("-- Turn off fast-math for host in SYCL backend")
    target_compile_options(${DIT_LIB} PRIVATE ${SYCL_COMPILE_OPTIONS})
endif()

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# see https://github.com/ggerganov/ggml/pull/682
add_definitions(-DGGML_MAX_NAME=128)

# deps
# Only add ggml if it hasn't been added yet
if (NOT TARGET ggml)
    add_subdirectory(ggml)
endif()

add_subdirectory(thirdparty)

target_link_libraries(${DIT_LIB} PUBLIC ggml zip)
target_include_directories(${DIT_LIB} PUBLIC . thirdparty)
target_compile_features(${DIT_LIB} PUBLIC cxx_std_11)


if (DIT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

