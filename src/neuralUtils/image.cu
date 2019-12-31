#include "image.hh"

Image::Image(size_t x_dim, size_t y_dim, bool hostOnly) :
    shape(x_dim, y_dim), deviceData(nullptr), hostData(nullptr),
    deviceAllocated(false), hostAllocated(false), hostOnly(hostOnly)
{ }

Image::Image(Shape shape, bool hostOnly) :
    Image(shape.x, shape.y, hostOnly)
{ }

void Image::allocateDeviceMemory() {
    if (!deviceAllocated) {
        cudaError_t ok;
        uint * deviceMemory = nullptr;


        ok = cudaMalloc(&deviceMemory, shape.x * shape.y * sizeof(uint));
        checkCudaErrors(ok);
        deviceData = std::shared_ptr<uint> (deviceMemory, [&](uint* ptr){ cudaFree(ptr); });
        deviceAllocated = true;
    }
}

void Image::allocateHostMemory() {
    if (!hostAllocated) {
        hostData = std::shared_ptr<uint> (new uint[shape.x*shape.y], [&](uint* ptr){ delete[] ptr; });
        hostAllocated = true;
    }
}

void Image::allocateMemory() {

    allocateHostMemory();
    
    if (!hostOnly) {
        allocateDeviceMemory();
    }
}

void Image::maybeAllocateMemory(Shape shape) {
    if (!deviceAllocated && !hostAllocated) {
        this->shape = shape;
        allocateMemory();
    } 
}

void Image::copyHostToDevice() {
    if (deviceAllocated && hostAllocated) {
        cudaError_t ok;
        ok = cudaMemcpy(deviceData.get(), hostData.get(), shape.x * shape.y * sizeof(uint), cudaMemcpyHostToDevice);
		checkCudaErrors(ok);
    } else {
        printf("Failed to copy from host to device... nothing initialized\n");
    }
}

void Image::copyDeviceToHost() {
    if (deviceAllocated && hostAllocated) {
        cudaError_t ok;
        ok = cudaMemcpy(
            hostData.get(), 
            deviceData.get(), 
            shape.x * shape.y * sizeof(uint), 
            cudaMemcpyDeviceToHost
        );

        checkCudaErrors(ok);

    } else {
        printf("Failed to copy from device to host... nothing initialized\n");
    }
}

uint& Image::operator[](const int index) {
	return hostData.get()[index];
}

const uint& Image::operator[](const int index) const {
	return hostData.get()[index];
}

