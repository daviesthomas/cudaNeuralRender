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
#include <thrust/iterator/transform_iterator.h>


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

const int COLOR_MASK_VAL = 2;
const float EPSILON = 0.0001;
const int MAX_STEPS = 100;

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

__device__ uint fragColor(int idx, const float * d_sdf, const uint* d_matcap, const uint offset) {
    float3 normal_eye;

    //printf("%d:%0.4f  %d:%0.4f  %d:%0.4f  %d:%0.4f %d:%0.4f  %d:%0.4f \n", idx,  d_sdf[idx], offset, d_sdf[offset],  offset+1, d_sdf[offset+1], offset+2, d_sdf[offset+2], offset+3, d_sdf[offset+3], offset+4, d_sdf[offset+4]);
    normal_eye.x = d_sdf[idx] - d_sdf[offset];
    normal_eye.y = d_sdf[offset + 1] - d_sdf[offset + 2];
    normal_eye.z = d_sdf[offset + 3] - d_sdf[offset + 4];
    normal_eye = normalize(normal_eye);
    
    int uvx = (normal_eye.x * 0.5 + 0.5) * 512;
    int uvy = (normal_eye.y * 0.5 + 0.5) * 512;
    
    int index = uvx * 512 + uvy;

    if (index >= 515*512) {
        printf("(%d %d): %d\n", uvx, uvy, index);
    }

    return d_matcap[index];//rgbaFloatToInt(make_float4(normal_eye.x, normal_eye.y, normal_eye.z,1.0));
}


__global__ void
singleMarch(
    uint* d_output,
    const uint* d_idSDFMap,
    const int batchOffset,
    uint* d_mask,
    const float* d_sdf,
    float* d_points,
    const float* d_ray,
    float* d_tfar,
    const uint* d_matcap,
    int imageW,
    int imageH
)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    int id = y*imageW + x;

    if (d_mask[id] == 0) return;    // masked out

    int inferenceIndex = d_idSDFMap[id];

    if (d_mask[id] >= COLOR_MASK_VAL) {
        // needs to be colored.
        d_output[id] = fragColor(inferenceIndex, d_sdf, d_matcap, d_mask[id] - COLOR_MASK_VAL);
        d_mask[id] = 0;
        return;
    }

    const float3 ray = getFloat3(d_ray, 3*id);
    float3 point = getFloat3(d_points, 3*id);

    const float tstep = -tanh(d_sdf[inferenceIndex]);
    d_tfar[id] -= tstep;

    if (d_tfar[id] <= 0) {
        d_mask[id] = 0;
        d_output[id] = rgbaFloatToInt(make_float4(0.0f));
        return;
    }

    // update point along ray
    point = point + ray*tstep;
    setFloat3(d_points, 3*id, point);

    // if close enough, we're done!
    if (tstep < EPSILON) {
        d_mask[id] = COLOR_MASK_VAL;     // prep for coloring!
    };
    
}

typedef thrust::tuple<int,float> IdxFloatTuple;
typedef thrust::device_ptr<uint> DeviceImgPtr;
typedef thrust::device_ptr<float> DeviceMatPtr;
typedef thrust::counting_iterator<int> IndexIter;
typedef thrust::tuple< IndexIter, DeviceMatPtr> IndexMatTuple;
typedef thrust::zip_iterator< IndexMatTuple > IndexMatZip;

struct isGt {
    uint value;

    isGt (uint val) : value(val) {};

    __device__ __host__
    uint operator()(const uint &x) {
        return (x > value);
    }
};


// since points is 3* size of mask. we need a way to know index of point we're looking at.
//          if x we want to look at mask[i/3]
//             y we want to look at mask[(i-1)/3]
//             z we want to look at mask[(i-2)/3]
// we can achieve this 'relatively' cheaply by creating a zip iterator with index.
struct needs_normals : public thrust::unary_function<IdxFloatTuple, bool>
{
    needs_normals(DeviceImgPtr mask, const DeviceImgPtr posIdxs, const DeviceImgPtr normIdxs, const uint offset) : mask(mask), posIdxs(posIdxs), normIdxs(normIdxs), offset(offset)  {}

    DeviceImgPtr mask;
    const uint offset;
    const DeviceImgPtr normIdxs;
    const DeviceImgPtr posIdxs;
 
    __device__
    bool operator()(const IdxFloatTuple& x) const 
    {
        uint idx = x.get<0>();
        float val = x.get<1>();
        uint mapIdx, maskIdx;
        bool isFirst = false;

        if (idx >= offset) {
            // normal point! (5 appended at end)
            mapIdx = ((idx-offset) - (idx-offset) %15)/15;
            maskIdx = (uint)normIdxs[mapIdx];
            isFirst = ((idx-offset) % 15) == 0;
        } else{
            // pos tracking points
            mapIdx = (idx - idx%3)/3;
            maskIdx = (uint)posIdxs[mapIdx];
        }

        // we update mask with idx of first normal sdf result
        if (mask[maskIdx] >= COLOR_MASK_VAL) {
            if (isFirst) {
                mask[maskIdx] = COLOR_MASK_VAL + idx/3;
            }
            return true;
        }
        return false;
    }
};

struct not_masked : public thrust::unary_function<IdxFloatTuple, bool>
{
    not_masked(const DeviceImgPtr mask) : mask(mask)  {}

    const DeviceImgPtr mask;

    __device__
    bool operator()(const IdxFloatTuple& x) const 
    {
        uint idx = x.get<0>();
        return (mask[(idx - idx%3)/3] > 0);
    }
};

struct normalEstimationPrep: public thrust::unary_function<IdxFloatTuple, IdxFloatTuple>
{
    normalEstimationPrep(const DeviceMatPtr points, const DeviceImgPtr posIdxs, const DeviceImgPtr normIdxs, const float eps, const int offset)
        : points(points), posIdxs(posIdxs), normIdxs(normIdxs), EPS(eps), offset(offset) {}

    const int modifier[15] = {
        -1,0,0,
         0,1,0,
         0,-1,0,
         0,0,1,
         0,0,-1
    };

    const DeviceMatPtr points;
    const DeviceImgPtr normIdxs;
    const DeviceImgPtr posIdxs;
    const float EPS;
    const uint offset;

    __device__
    IdxFloatTuple operator()(const IdxFloatTuple& x) const
    {
        uint idx = x.get<0>();
        
        uint pointIdx;
        int modType;
        float a;

        if (idx >= offset) {
            pointIdx = normIdxs[((idx-offset) - (idx-offset)%15)/15]*3 + idx%3;
            modType = modifier[(idx-offset)%15];
            a = points[pointIdx];
        } else {
            modType = (idx%3 == 0) ? 1 : 0;
            a = x.get<1>();
        }

        return thrust::make_tuple(x.get<0>(), a + modType*EPS);
    }
};


void printCrap(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch) {
    idSDFMap.copyDeviceToHost();
    stepMask.copyDeviceToHost();
    points.copyDeviceToHost();
    batch.copyDeviceToHost();

    for (int i = 0; i < idSDFMap.size(); i ++) {
        int ptIdx = idSDFMap[i];

        if (stepMask[i] > 0) {
            printf("%d-%d:%d:  (%f,%f,%f)&(%f,%f%f)", i, stepMask[i], idSDFMap[i], points[i*3], points[i*3 + 1], points[i*3 + 2],batch[ptIdx*3], batch[ptIdx*3 + 1], batch[ptIdx*3 + 2]);
            if (stepMask[i] >= COLOR_MASK_VAL) {
                printf(" <---");
            }
            printf("\n");
        }

        
    }
    printf("BATCH: \n");
    for (int i = 0; i < batch.size()-2; i += 3) {
        
        printf("\t %d: (%f, %f, %f) \n", i/3, batch[i],batch[i+1],batch[i+2] );
    }
    std::cout << "\n\n";
}


/*
Function to prep inputs for inference (gemm) by copying points to a batch buffer according to supplied mask.
Additionally, if mask == COLOR_MASK_VAL, points for estimating the normal at surface are appended to batch.

idSDFMap - Positions of points in batch (and resultant SDF array) is stored in idSDFMap
stepMask - binary mask for which positions in image still require marching
points   - point matrix used to track positions of all rays
batch    - 'vector' like object, dymanically sized for number of inferences in a batch
*/
int formatInferenceReqs(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch, thrust::device_vector<uint>& tempIdxs) {
    // This function needs heavy refactoring... Should really be done in a single kernel...

    /*
    // CHANGE TO THIS
    // d_mask == 1 if step, 6 if normal. required.
    thrust::exclusive_scan(
        thrust::device,
        d_mask, 
        d_mask + stepMask.size(), 
        d_idSDFMap,
        0,
        thrust::plus<uint>()
    );

    // custom kernel for copying to batch
    // 
    createBatch<<<blocks,threadsperblock>>> (d_idSDFMap, d_mask, d_points, d_batch);
    // if d_mask[idx] > 0
    //  batchIdx = d_idSDFMap[idx]
    //  if d_mask[idx] == 1
    //    d_batch[batchIdx:batchIdx+3] = d_point[idx:idx +3] 
    //  if d_mask[idx] == 6
    //    d_batch[batchIdx:batchIdx+3] = modEps( d_points[idx: idx+3], 0, 1)  
    //    d_batch[batchIdx+3:batchIdx+6] = modEps( d_points[idx: idx+3], 0, -1) 
    //    d_batch[batchIdx+6:batchIdx+9] = modEps( d_points[idx: idx+3], 1, 1) 
    //    d_batch[batchIdx+9:batchIdx+12] = modEps( d_points[idx: idx+3], 1, -1) 
    //    d_batch[batchIdx+15:batchIdx+18] = modEps( d_points[idx: idx+3], 2, 1) 
    //    d_batch[batchIdx+18:batchIdx+21] = modEps( d_points[idx: idx+3], 2, -1) 

    */

    
    int numColorReqs = 0;
    int numPointReqs = 0;

    // create thrust ptrs for compat
    DeviceImgPtr d_mask(stepMask.deviceData.get());                 // image mask. 0 if no calc needed, 1 if stepping, 2 if color calc.
    DeviceImgPtr d_idSDFMap(idSDFMap.deviceData.get());             // given id return index of sdf val and points (3*index gives index of x val, +1 == y)
    DeviceImgPtr d_normalIdSDFMap(idSDFMap.deviceData.get());       // given id return index of second normal point...
    DeviceMatPtr d_points(points.deviceData.get());                 // position tracker for all possible rays in image (constant size H*W*3)
    DeviceMatPtr d_batch(batch.deviceData.get());                   // 'vector' of positions that need sdf inference. (dynamically sized!)

    thrust::counting_iterator<int> idxFirst(0);
    IndexMatZip firstPoint = thrust::make_zip_iterator(thrust::make_tuple(idxFirst, d_points));
    IndexMatZip firstBatch = thrust::make_zip_iterator(thrust::make_tuple(idxFirst, d_batch));

    // Copy points required for ray marching (that need inference)
    numPointReqs = (thrust::copy_if(
        thrust::device,
        firstPoint,
        firstPoint + points.size(),
        firstBatch,
        not_masked(d_mask)    
    ) - firstBatch)/3;
    
    if (numPointReqs > 0) {

        DeviceImgPtr colorIdxs(tempIdxs.data());

        // get indeces of pts marked to be colored.
        numColorReqs = thrust::copy_if(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(stepMask.size()),
            d_mask,
            colorIdxs,
            isGt(COLOR_MASK_VAL-1)
        ) - colorIdxs;


        // update batch with additional reqs for normal estimation
        if (numColorReqs > 0) {
            DeviceImgPtr pointIdxs(tempIdxs.data() + numColorReqs);

            int totalPointReqs = numPointReqs + numColorReqs*5;

            // get indeces of non-zero items in mask
            thrust::copy_if(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(stepMask.size()),
                d_mask,
                pointIdxs,
                isGt(0)
            );
    
            // Now we append pts for normal estimation to the batch matrix.        
            thrust::transform_if(
                thrust::device,
                firstBatch,
                firstBatch + totalPointReqs*3,
                firstBatch,
                normalEstimationPrep(d_points,  pointIdxs, colorIdxs, EPSILON, numPointReqs*3),              // op 
                needs_normals(d_mask, pointIdxs, colorIdxs, numPointReqs*3)   // predicate
            );
        }

        // Prefix Scan...counting only 1s (we want to map to)
        thrust::transform_exclusive_scan(
            thrust::device,
            d_mask, 
            d_mask + stepMask.size(), 
            d_idSDFMap,
            isGt(0),
            0,
            thrust::plus<uint>()
        );
    }

    int batchSize = numPointReqs + numColorReqs*5;
    batch.shape.y = batchSize;

    //printCrap(idSDFMap, stepMask, points, batch);

    return numPointReqs;
}


Matrix points;
Matrix batch;
Matrix ray;
Matrix far;
Image stepMask;
Image idSDFMap;
thrust::device_vector<uint> *tempIdxs = NULL;    
int prevImageSize = -1;

void allocateBuffers(const int imageW, const int imageH) {
    int imageSize = imageW*imageH;

    points = Matrix(Shape(3, imageSize));
    batch = Matrix(Shape(3*6, imageSize));
    ray = Matrix(Shape(3, imageSize));
    far = Matrix(Shape(1, imageSize));
    stepMask = Image(Shape(imageH, imageW));
    idSDFMap = Image(Shape(imageH, imageW));

    if (tempIdxs != NULL){
        delete tempIdxs;
    }
    tempIdxs = new thrust::device_vector<uint>(imageSize*2); 

    // allocate them
    points.allocateMemory();
    ray.allocateMemory();
    far.allocateMemory();
    stepMask.allocateMemory();
    idSDFMap.allocateMemory();
    batch.allocateMemory();
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
    int imageSize = imageH*imageW;

    if (imageSize != prevImageSize) {
        allocateBuffers(imageW, imageH);
        prevImageSize = imageSize;
    }
    
    Matrix sdf;

    initMarcher<<<gridSize, blockSize>>> (
        d_output, 
        stepMask.deviceData.get(), 
        points.deviceData.get(), 
        ray.deviceData.get(),
        far.deviceData.get(),
        imageW, 
        imageH, 
        MAX_STEPS
    );

    // remove masked pixel (and reduce batch size if possible)
    int numPointReqs = formatInferenceReqs(
        idSDFMap,
        stepMask,
        points,
        batch,
        *tempIdxs
    );

    // march all rays simultaneossly. (so we can utilize batched gemm optimizations)
    for (int i = 0; i < MAX_STEPS; i ++) {   
        
        if (batch.shape.y == 0) {
            //nothing to do!
            break;
        }

        // infer all points required
        sdf = nn.forward(batch, imageSize*6);  //the x6 is too ensure memory exists if entire image needs to be colored.

        // take step, updating mask, points, and ray position (tfar)
        singleMarch<<<gridSize, blockSize>>>(
            d_output,
            (const uint *)idSDFMap.deviceData.get(), 
            numPointReqs,  //batchOffset
            stepMask.deviceData.get(),
            (const float *)sdf.deviceData.get(), 
            points.deviceData.get(), 
            (const float *)ray.deviceData.get(),
            far.deviceData.get(),
            (const uint *)matcap,
            imageW, 
            imageH
        );

        // remove masked pixel (and reduce batch size if possible)
        numPointReqs = formatInferenceReqs(
            idSDFMap,
            stepMask,
            points,
            batch,
            *tempIdxs
        );
        
    }
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
