#pragma once

#include "shape.hh"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <memory>

class Matrix {
    public:
        bool deviceAllocated;
        bool hostAllocated;

        void allocateDeviceMemory();
        void allocateHostMemory();

    public:
        Shape shape;

        std::shared_ptr<float> deviceData;
        std::shared_ptr<float> hostData;

        Matrix(size_t x_dim = 1, size_t y_dim = 1);
        Matrix(Shape shape);

        void allocateMemory();
        void maybeAllocateMemory(Shape shape);

        void copyHostToDevice();
        void copyDeviceToHost();

        // operator overrides for array like access
        float& operator[](const int index);
	    const float& operator[](const int index) const;
};