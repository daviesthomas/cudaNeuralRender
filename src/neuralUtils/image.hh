#pragma once

#include "shape.hh"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <memory>

class Image {
    private:
        bool deviceAllocated;
        bool hostAllocated;
        bool hostOnly = false;

        void allocateDeviceMemory();
        void allocateHostMemory();

    public:
        Shape shape;

        std::shared_ptr<uint> deviceData;
        std::shared_ptr<uint> hostData;

        Image(size_t x_dim = 1, size_t y_dim = 1, bool hostOnly = false);
        Image(Shape shape, bool hostOnly = false);

        void allocateMemory();
        void maybeAllocateMemory(Shape shape);

        void copyHostToDevice();
        void copyDeviceToHost();

        // operator overrides for array like access
        uint& operator[](const int index);
	    const uint& operator[](const int index) const;
};