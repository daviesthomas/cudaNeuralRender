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
    float *d_pos,
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

    d_output[id] = rgbaFloatToInt(make_float4(0.0f));

    // no need to march if ray never hits sphere!
    if (!hit) {
        d_mask[id] = 0;
        d_output[id] = rgbaFloatToInt(make_float4(0.0f));
        setFloat3(d_pos, 3*id, make_float3(0.0));
        return;
    }

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // start ray at edge of bounds
    float3 pos = eyeRay.o + eyeRay.d*tnear;

    //store starting position
    setFloat3(d_pos, 3*id, pos);
    //store ray (to skip comps going forward)
    setFloat3(d_ray, 3*id, eyeRay.d);
    //store init tlim to tfar.
    d_tfar[id] = tfar;
    // init mask to max steps
    d_mask[id] = maxSteps;
}

__global__ void
singleMarch(
    uint* d_output,
    uint* d_stepMask,
    const float* d_sdf,
    float* d_pos,
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

    if (d_stepMask[id] == 0) return;

    const float3 ray = getFloat3(d_ray, 3*id);
    float3 pos = getFloat3(d_pos, 3*id);

    const float tstep = -tanh(d_sdf[id]);
    const float tlim = d_tfar[id] + tstep;

    float3 newPos = pos + ray*tstep;
    //printf("pos: (%0.4f, %0.4f, %0.4f) ---|%0.4f|--> (%0.4f,%0.4f,%0.4f) \n", pos.x,pos.y,pos.z,tstep,newPos.x, newPos.y, newPos.z);
    pos = pos + ray*tstep;

    // if close enough, we're done!
    if (tstep < EPSILON) {
        //printf("Done --> (%f, %f, %f): %f\n",pos.x,pos.y,pos.z, tstep);
        //mask out in step mask to avoid comps
        d_stepMask[id] = 0;
        //now color!
        d_output[id] = rgbaFloatToInt(make_float4(1.0f));
    };

    if (tlim <=0) {
        //we never hit anything within box!
        d_stepMask[id] = 0;
        d_output[id] = rgbaFloatToInt(make_float4(0.0f));
    }

    d_tfar[id] -= tstep;
    d_stepMask[id] -= 1;
    setFloat3(d_pos, 3*id, pos);
}

extern "C"
void render_kernel(
    dim3 gridSize,
    dim3 blockSize,
    uint *d_output, 
    uint imageW, 
    uint imageH,
    NeuralNetwork& nn
) {
    printf("Start Render\n");
    
    const int MaxSteps = 60;

    Matrix pos = Matrix(Shape(3, imageH*imageW));
    Matrix ray = Matrix(Shape(3, imageH*imageW));
    Matrix far = Matrix(Shape(1, imageH*imageW));

    Matrix sdf;
    Image stepMask = Image(Shape(imageH, imageW));

    // allocate them
    pos.allocateMemory();
    ray.allocateMemory();
    far.allocateMemory();
    stepMask.allocateMemory();
    
    initMarcher<<<gridSize, blockSize>>> (
        d_output, 
        stepMask.deviceData.get(), 
        pos.deviceData.get(), 
        ray.deviceData.get(),
        far.deviceData.get(),
        imageW, 
        imageH, 
        MaxSteps
    );

    // march all rays simultaneossly. (so we can utilize batched gemm optimizations)
    for (int i = 0; i < MaxSteps; i ++) {
        // marcher initializes the step mask and position query array.
        //TODO: we currently just infer the max batch size on each iteration... this should not be the case.
        //        mask array should be used to reduce size of inference as pixels are deemed unnesary!
        sdf = nn.forward(pos);

        // get steps kernel takes a single step and updates d_mask updated step count.
        singleMarch<<<gridSize, blockSize>>>(
            d_output,
            stepMask.deviceData.get(), 
            (const float *)sdf.deviceData.get(), 
            pos.deviceData.get(), 
            (const float *)ray.deviceData.get(),
            far.deviceData.get(),
            imageW, 
            imageH
        );
    }
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
