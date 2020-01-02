#include "image.hh"

float saturatef(float x)
{
    return min(1.0, max(x, 0.0));
}

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

bool Image::loadPNG(std::string filename) {
    std::vector<unsigned char> png; 
    unsigned w, h;

    // get h and w to allocate memory
    unsigned error = lodepng::decode(png, w, h, filename);

    if (error) {
        std::cout << "Error reading png: " << lodepng_error_text(error) << std::endl;
        return 0;
    }

    maybeAllocateMemory(Shape((int)w,(int)h));

    float r, g, b, a;

    // copy into host memory.
    for (int i = 0 ; i < (png.size()/4); i ++) {
        r = saturatef(png[i*4]);   // clamp to [0.0, 1.0]
        g = saturatef(png[i*4+1]);
        b = saturatef(png[i*4+2]);
        a = saturatef(png[i*4+3]);
        hostData.get()[i] = (uint(a*255)<<24) | (uint(b*255)<<16) | (uint(g*255)<<8) | uint(r*255);
    }

    // copy into device memory (should be constant...)
    copyHostToDevice();

    return 1;
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

