/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"
#include "neuralUtils/image.hh"

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


typedef unsigned int  uint;
typedef unsigned char uchar;

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};


// intersect ray with a box
__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}


// intersect ray with a box
__device__
int intersectSphere(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

// given a point return distance to surface
__device__ float distanceToSurface(float3 pos)
{
	const float radius = 0.5f;
    return length(pos)-radius;
}

//given a point on surface, calc & retrun normal.
__device__ float3 fragNormal(float3 p)
{
    float3 n;
    float3 a,b;
    const float EPSILON = 0.00001;

    a.x = p.x + EPSILON;
    a.y = p.y;
    a.z = p.z;

    b.x = p.x - EPSILON;
    b.y = p.y;
    b.z = p.z;

    n.x = distanceToSurface(a) - distanceToSurface(b);

    a.x = p.x;
    a.y = p.y + EPSILON;
    b.x = p.x;
    b.y = p.y - EPSILON;

    n.y = distanceToSurface(a) - distanceToSurface(b);

    a.y = p.y;
    a.z = p.z + EPSILON;
    b.y = p.y;
    b.z = p.z - EPSILON;

    n.z = distanceToSurface(a) - distanceToSurface(b);


    return normalize(n);
}

__device__ 
float3 getFloat3(const float* D, int id) {
    return make_float3(
        D[id],
        D[id + 1],
        D[id + 2]
    );
}

__device__ 
void setFloat3(float* D, int id, float3 f) {
    D[id] = f.x;
    D[id + 1] = f.y;
    D[id + 2] = f.z;
}

__global__ void
initMarcher(
    uint *d_output, 
    uint *d_mask,
    float *d_points,
    float *d_ray,
    float *d_tfar,
    uint imageW, 
    uint imageH, 
    int maxSteps
)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    const float3 boxMin = make_float3(-0.5f, -0.5f, -0.5f);
    const float3 boxMax = make_float3(0.5f, 0.5f, 0.5f);

    int id = y*imageW + x;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    // no need to march if ray never hits bounding primitive
    if ( !hit ) {
        d_mask[id] = 0;
        d_output[id] = rgbaFloatToInt(make_float4(0.0f));
        return;
    }

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // start ray at edge of bounds
    float3 point = eyeRay.o + eyeRay.d*tnear;

    //store starting position
    setFloat3(d_points, 3*id, point);
    
    //store ray (to skip comps going forward)
    setFloat3(d_ray, 3*id, eyeRay.d);
    
    //store tfar. (decremented as we march the ray), exit condition
    d_tfar[id] = tfar;

    // Stencil update
    d_mask[id] = 1;
}

__device__ uint fragColor(int idx, const float * d_sdf) {
    return rgbaFloatToInt(make_float4(1.0f));
}


__global__ void
singleMarch(
    uint* d_output,
    const uint* d_idSDFMap,
    uint* d_mask,
    const float* d_sdf,
    float* d_points,
    const float* d_ray,
    float* d_tfar,
    int imageW,
    int imageH
)
{
    const float EPSILON = 0.00001;

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    int id = y*imageW + x;

    if (d_mask[id] == 0) return;    // masked out

    int inferenceIndex = d_idSDFMap[id];

    const float3 ray = getFloat3(d_ray, 3*id);
    float3 point = getFloat3(d_points, 3*id);

    const float tstep = -tanh(d_sdf[inferenceIndex]);
    d_tfar[id] -= tstep;

    if (d_tfar[id] <= 0) {
        d_mask[id] = 0;
        d_output[id] = rgbaFloatToInt(make_float4(0.0f));
        return;
    }

    //printf("Im moving: %d: (%f,%f,%f):%f  %d %f\n", id, point.x, point.y, point.z, tstep, d_mask[id], d_tfar[id]);
    // update point along ray
    point = point + ray*tstep;
    //printf("moved: %d: (%f,%f,%f):%f \n", id, point.x, point.y, point.z, tstep);
    setFloat3(d_points, 3*id, point);

    // if close enough, we're done!
    if (tstep < EPSILON) {
        d_mask[id] = 0; 
        d_output[id] = fragColor(inferenceIndex, d_sdf);
    };
    
}

struct countOnlyOne {
    __device__ __host__
    uint operator()(const uint &x) {
        return (x == 1);
    }
};

struct onlyCountOverOne {
    __device__ __host__
    uint operator()(const uint &x) {
        return (x > 1) ? x : 0;
    }
};

typedef thrust::tuple<int,float> IdxFloatTuple;
typedef thrust::device_ptr<uint> DeviceImgPtr;
typedef thrust::device_ptr<float> DeviceMatPtr;
typedef thrust::counting_iterator<int> IndexIter;
typedef thrust::tuple< IndexIter, DeviceMatPtr> IndexMatTuple;
typedef thrust::zip_iterator< IndexMatTuple > IndexMatZip;


// since points is 3* size of mask. we need a way to know index of point we're looking at.
//          if x we want to look at mask[i/3]
//             y we want to look at mask[(i-1)/3]
//             z we want to look at mask[(i-2)/3]
// we can achieve this 'relatively' cheaply by creating a zip iterator with index.
struct not_masked : public thrust::unary_function<IdxFloatTuple, bool>
{
    not_masked(const uint* a_mask) : d_mask(a_mask) {}
    const uint* d_mask;
 
    __device__
    bool operator()(const IdxFloatTuple& x) const 
    {
        uint idx = x.get<0>();
        uint maskIdx = (idx - idx%3)/3;
        //printf("idx: %d maskIdx: %d  mask: %d \n", idx, maskIdx, d_mask[maskIdx]);
        return d_mask[maskIdx];
    }
};

void printCrap(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch) {
    idSDFMap.copyDeviceToHost();
    stepMask.copyDeviceToHost();
    points.copyDeviceToHost();
    batch.copyDeviceToHost();

    for (int i = 0; i < idSDFMap.size(); i ++) {
        int ptIdx = idSDFMap[i];

        printf("%d-%d:%d:  (%f,%f,%f)&(%f,%f%f)", i, stepMask[i], idSDFMap[i], points[i*3], points[i*3 + 1], points[i*3 + 2],batch[ptIdx*3], batch[ptIdx*3 + 1], batch[ptIdx*3 + 2]);
        if (stepMask[i]) {
            printf(" <---");
        }
        printf("\n");
    }
    std::cout << "\n\n";
}

int formatInferenceReqs(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch, int first = false) {
    //printf("........new format....\n");
    // create thrust ptrs for compat
    DeviceImgPtr d_mask(stepMask.deviceData.get());                 // image mask. 0 if no calc needed, 1 if stepping, 2 if color calc.
    DeviceImgPtr d_idSDFMap(idSDFMap.deviceData.get());             // given id return index of sdf val and points (3*index gives index of x val, +1 == y)
    DeviceMatPtr d_points(points.deviceData.get());                // 'vector' of points [xyzxyzxyzxyz...] 
    DeviceMatPtr d_batch(batch.deviceData.get());
    //thrust::device_ptr<uint> d_idNormalSDFMap(idNormalSDFMap.deviceData.get()); // map that points to first sdf value for normal calculation (note this should always be << d_mask!)

    //printCrap(idSDFMap, stepMask, points,batch);

    thrust::counting_iterator<int> idxFirst(0);
    thrust::counting_iterator<int> idxLast = idxFirst + points.size();

    IndexMatZip firstPoint = thrust::make_zip_iterator(thrust::make_tuple(idxFirst, d_points));
    IndexMatZip lastPoint = thrust::make_zip_iterator(thrust::make_tuple(idxLast, d_points + points.size()));
    IndexMatZip firstBatch = thrust::make_zip_iterator(thrust::make_tuple(idxFirst, d_batch));

    // overwrite point array with unmasked data (effecitvely squashing unneeded points.)
    int batchSize = (thrust::copy_if(
        thrust::device,
        firstPoint,
        lastPoint,
        firstBatch,
        not_masked(stepMask.deviceData.get())    
    ) - firstBatch)/3;

    // now we want to get all instances of mask == 2 (indicates color request.)
    // lets just increase size of points array. 
    // each pixel can have 6 [x,y,z] pts allocated... 
        // access to pos tracker via i*6*3
        // access to normals via i*6*3 + 3*n (where n is index of normal to access.)
    // then we increment idSDFMap by +1 if pos, or +6 if normal calc. Normals should be infrequent relative to pos, so not big deal.
    
    // Prefix Scan...counting only 1s
    thrust::transform_exclusive_scan(
        thrust::device,
        d_mask, 
        d_mask + stepMask.size(), 
        d_idSDFMap,
        countOnlyOne(),
        0,
        thrust::plus<uint>()
    );

    //printCrap(idSDFMap, stepMask, points, batch);
    
    batch.shape.y = batchSize; // this is unsafe and should be checked.

    return batchSize;
}

extern "C"
void render_kernel(
    dim3 gridSize,
    dim3 blockSize,
    uint *d_output, 
    uint imageW, 
    uint imageH,
    NeuralNetwork& nn,
    uint *matcap
) {

    const int MaxSteps = 60;

    int imageSize = imageH*imageW;

    Matrix points = Matrix(Shape(3, imageSize));
    Matrix batch = Matrix(Shape(3, imageSize));
    Matrix ray = Matrix(Shape(3, imageSize));
    Matrix far = Matrix(Shape(1, imageSize));
    Matrix sdf;
    
    Image stepMask = Image(Shape(imageH, imageW));
    Image idSDFMap = Image(Shape(imageH, imageW));

    // allocate them
    points.allocateMemory();
    ray.allocateMemory();
    far.allocateMemory();
    stepMask.allocateMemory();
    idSDFMap.allocateMemory();
    batch.allocateMemory();
    
    initMarcher<<<gridSize, blockSize>>> (
        d_output, 
        stepMask.deviceData.get(), 
        points.deviceData.get(), 
        ray.deviceData.get(),
        far.deviceData.get(),
        imageW, 
        imageH, 
        MaxSteps
    );

    int pointsSize = points.size();

    // remove masked pixel (and reduce batch size if possible)
    int batchSize = formatInferenceReqs(
        idSDFMap,
        stepMask,
        points,
        batch,
        true
    );

    // march all rays simultaneossly. (so we can utilize batched gemm optimizations)
    for (int i = 0; i < MaxSteps; i ++) {   
        
        if (batchSize == 0) break;

        // infer all points required
        sdf = nn.forward(batch, imageSize);  

        // take step, updating mask, points, and ray position (tfar)
        singleMarch<<<gridSize, blockSize>>>(
            d_output,
            (const uint *)idSDFMap.deviceData.get(), 
            stepMask.deviceData.get(),
            (const float *)sdf.deviceData.get(), 
            points.deviceData.get(), 
            (const float *)ray.deviceData.get(),
            far.deviceData.get(),
            imageW, 
            imageH
        );

        // remove masked pixel (and reduce batch size if possible)
        int batchSize = formatInferenceReqs(
            idSDFMap,
            stepMask,
            points,
            batch,
            false
        );
        
    }
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
