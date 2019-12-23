cmake_minimum_required(VERSION 2.8)

project(sdfRenderer)

find_package(CUDA QUIET REQUIRED)
find_package(GLUT REQUIRED)

include_directories(
    kernels
    common
)

cuda_add_executable(
    sdfRenderer
    volumeRender_kernel.cu
    volumeRender.cpp
)

include_directories( ${GLUT_INCLUDE_DIRS} )
target_link_libraries(sdfRenderer -lglut -lGL)