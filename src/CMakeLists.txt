project(sdfRenderer LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(HighFive 2.1 REQUIRED)

if (CUDA_VERSION VERSION_LESS 9.2)
  message(FATAL_ERROR "CUDA 9.2+ Required, Found ${CUDA_VERSION}.")
endif()

# Special policy introduced in CMake 3.13
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif() 

# default to release (2Fast2Furious)
if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

if (CUDA_VERSION VERSION_LESS 10.1)
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT OFF)
else()
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT ON)
endif()

include_directories(
  layers
  common
  neuralUtils
  ${CUDA_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIRS}
  ../submodules/cutlass/include
  ../submodules/HighFive/include
)

cuda_add_executable(
    sdfRenderer
    volumeRender_kernel.cu
    volumeRender.cpp
)

set_property(TARGET sdfRenderer PROPERTY CUDA_STANDARD 11)

target_link_libraries(sdfRenderer ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${CUDA_LIBRARIES} HighFive)

cuda_add_executable(
    neuralNet
    neuralNetwork.cu
    layers/denseLayer.cu
    neuralUtils/matrix.cu 
    neuralUtils/shape.cu
    main.cpp
)

set_property(TARGET neuralNet PROPERTY CUDA_STANDARD 11)

target_link_libraries(neuralNet ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${CUDA_LIBRARIES} HighFive)