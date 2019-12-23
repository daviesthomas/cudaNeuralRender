cmake_minimum_required(VERSION 3.12)

project(sdfRenderer LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

if (CUDA_VERSION VERSION_LESS 9.2)
  message(FATAL_ERROR "CUDA 9.2+ Required, Found ${CUDA_VERSION}.")
endif()

add_subdirectory(submodules/cutlass)

include_directories(
    kernels
    common
    ${CUDA_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    submodules/cutlass/include
)

cuda_add_executable(
    sdfRenderer
    kernels/volumeRender_kernel.cu
    volumeRender.cpp
)

target_link_libraries(sdfRenderer ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${CUDA_LIBRARIES})