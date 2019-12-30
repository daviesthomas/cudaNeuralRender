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

__global__ void relu(float* Z, float* A,
    int Z_x_dim, int Z_y_dim) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = fmaxf(Z[index], 0);
    }
}

__global__ void denseForward(
    float* W, 
    float* A,
    float* Z, 
    float* b,
    int W_x_dim, int W_y_dim,
    int A_x_dim, int A_y_dim, int activation) 
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim) {
        for (int i = 0; i < W_x_dim; i++) {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

__device__ 
float forwardInfer(
    float* weights, 
    float* biases, 
    float* dims, 
    float* A,
    float* Z,
    int numLayers
) 
{    
    // position in mem for each layers weights and biases.
    int wPos = 0;
    int bPos = 0;
    
    for (int i = 0; i < numLayers; i ++) {
        int M = int(dims[i*3]);
        int N = int(dims[i*3+1]);
        int K = int(dims[i*3+2]);

        // just for reference...
        //int Wx = M;
        //int Wy = K;
        //int Ax = N;
        //int Ay = K;
        //int bx = Ax;
        //int by = Wy;
        //int Zx = Ax;
        //int Zy = Wy;

        dim3 blockSize(4, 4);
        dim3 threadsPerBlock(	(M + blockSize.x - 1) / blockSize.x,
                                (K + blockSize.y - 1) / blockSize.y);

        //Dynamic parallelism. 
        denseForward<<<blockSize,threadsPerBlock>>>(
            weights + wPos, 
            A, 
            Z, 
            biases + bPos,
            M,
            K,
            N,
            K,
            0
        );

        cudaDeviceSynchronize();

        if (i < (numLayers - 1)){
            relu<<<1,K*N>>>(Z, A, N, K);
            cudaDeviceSynchronize();
            // increment pointer to next layer of weights
            wPos += M*K;
            bPos += N*K;
        }
    }
    float Y = tanh(Z[0]);

    return Y;
}

//simple kernel for testing forward inference.
__global__ void 
inferTest(
    float* weights, 
    float* biases, 
    float* dims,
    float* A,
    float* Z,
    int numLayers
) {
    int id = 32;

    A[id+0] = static_cast<float>(-0.245570);
    A[id+1] = static_cast<float>(-0.429746);
    A[id+2] = static_cast<float>( -0.245570);
    float tstep = forwardInfer(weights, biases, dims, A+id, Z+id, numLayers);
    printf("tstep: %f \n", tstep);
}


__global__ void
d_render(
    uint *d_output, 
    uint imageW, 
    uint imageH, 
    float* weights, 
    float* biases, 
    float* dims,
    float* A,
    float* Z,
    int numLayers)
{
    const int maxSteps = 60;
    
    const float EPSILON = 0.00001;
    const float3 boxMin = make_float3(-0.5f, -0.5f, -0.5f);
    const float3 boxMax = make_float3(0.5f, 0.5f, 0.5f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    int id = x*imageW + y;

    if ((x >= imageW) || (y >= imageH)) return;

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

    // no need to march if ray never hits sphere!
    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    float tstep;
    float t = tnear;
    // start ray at edge of bounds
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step;

    // march the ray!
    for (int i=0; i<maxSteps; i++)
    {
        // get dist to surface
        A[id*32+0] = static_cast<float>(pos.x);
        A[id*32+1] = static_cast<float>(pos.y);
        A[id*32+2] = static_cast<float>(pos.z);

        //tstep = -forwardInfer(weights, biases, dims, A + id*32, Z + id*32, numLayers);
        tstep = distanceToSurface(pos);

        // if close enough, we're done!
        if (tstep < EPSILON) break;
        // step along ray
        step = eyeRay.d*tstep;
        t += tstep;
        // if past bounding box, we're done!
        if (t > tfar) break;
        // step ray for next iter
        pos += step;
    }

    float4 col; ;
    if (tstep < EPSILON) {
        // set color based on normals! (we'll later use matcap to look up in iamge...)

        pos += eyeRay.d * tstep;
        float3 normal = fragNormal(pos);
        col.x = normal.x;
        col.y = normal.y;
        col.z = normal.z;
        col.w = 1.0;

    } else {
        // either left the box OR reached max steps.
        col = make_float4(0.2f);
    }

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(col);
}

extern "C"
void render_kernel(
    dim3 gridSize, 
    dim3 blockSize, 
    uint *d_output, 
    uint imageW, 
    uint imageH,
    float* weights, 
    float* biases, 
    float* dims,
    float* A,
    float* Z,
    int numLayers
)
{

    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, weights, biases, dims, A, Z, numLayers);
}

extern "C"
void sdf_kernel(
    float* weights, 
    float* biases, 
    float* dims,
    float* A,
    float* Z,
    int numLayers
)
{
    inferTest<<<1,1>>>(weights, biases, dims, A, Z, numLayers);
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
