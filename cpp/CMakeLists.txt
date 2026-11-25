cmake_minimum_required(VERSION 3.10)
project(trt_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ---- 依赖路径：TX2 上 TensorRT 头文件在 /usr/include/aarch64-linux-gnu ----
find_package(OpenCV REQUIRED)

# 头文件目录：优先系统路径（TX2 的 NvInfer.h 不在 /usr/include 根下）
set(TRT_INC_HINTS
  /usr/include/aarch64-linux-gnu
  /usr/include/x86_64-linux-gnu
  /usr/include
  /usr/local/cuda/include
)

find_path(NVINFER_INCLUDE_DIR NvInfer.h HINTS ${TRT_INC_HINTS})
if(NOT NVINFER_INCLUDE_DIR)
  message(FATAL_ERROR "找不到 NvInfer.h，请确认已安装 TensorRT 开发包")
endif()
message(STATUS "NvInfer.h: ${NVINFER_INCLUDE_DIR}")

find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64)
if(NOT NVINFER_LIB OR NOT CUDART_LIB)
  message(FATAL_ERROR "找不到 nvinfer 或 cudart")
endif()
message(STATUS "nvinfer: ${NVINFER_LIB}")
message(STATUS "cudart : ${CUDART_LIB}")

# ---- 目标 ----
add_executable(trt_demo
  src/main.cpp
  src/trt_engine.cpp
)

target_include_directories(trt_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${NVINFER_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(trt_demo PRIVATE
  ${NVINFER_LIB}
  ${CUDART_LIB}
  ${OpenCV_LIBS}
)
