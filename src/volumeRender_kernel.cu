#ifndef _NEURALRENDER_KERNEL_CU_
#define _NEURALRENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"
#include "neuralUtils/image.hh"

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/scan.h>

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

const int COLOR_MASK_VAL = 6;
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

__device__ uint fragColor(int idx, const float * d_sdf, const uint* d_matcap) {
    float3 normal_eye;

    //printf("%d:%0.4f  %d:%0.4f  %d:%0.4f  %d:%0.4f %d:%0.4f  %d:%0.4f \n", idx,  d_sdf[idx], offset, d_sdf[offset],  offset+1, d_sdf[offset+1], offset+2, d_sdf[offset+2], offset+3, d_sdf[offset+3], offset+4, d_sdf[offset+4]);
    normal_eye.x = d_sdf[idx] - d_sdf[idx+1];
    normal_eye.y = d_sdf[idx+2] - d_sdf[idx + 3];
    normal_eye.z = d_sdf[idx + 4] - d_sdf[idx + 5];
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
        d_output[id] = fragColor(inferenceIndex, d_sdf, d_matcap);
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


__global__
void createBatch (float* d_batch, const uint* d_idSDFMap, const uint* d_mask, const float* d_points, const int imageW, const int imageH) {
    const int modifier[] = {
         1, 0, 0,
        -1, 0, 0,
         0, 1, 0,
         0,-1, 0,
         0, 0, 1,
         0, 0,-1
    };
    
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    int idx = y*imageW + x;
    uint maskVal = d_mask[idx];

    if (maskVal == 0) return;

    // index of where to store points in batch
    uint batchIdx = d_idSDFMap[idx]*3;

    // set points according to mask val
    // mask val == 1 for step request (1 points inference)
    // mask val == 6 for normal request (6 points inference)
    if (maskVal == 1) {
        d_batch[batchIdx ] = d_points[idx*3];
        d_batch[batchIdx + 1] = d_points[idx*3 + 1];
        d_batch[batchIdx + 2] = d_points[idx*3 + 2];
    } else {
        // add all points for normal estimation [x+eps, y, z] [x-eps, y, z] ....
        for (int i = 0; i < 3*maskVal; i += 3) {
            d_batch[batchIdx + i]     = d_points[idx*3]     + modifier[i]*EPSILON;
            d_batch[batchIdx + i + 1] = d_points[idx*3 + 1] + modifier[i+1]*EPSILON;
            d_batch[batchIdx + i + 2] = d_points[idx*3 + 2] + modifier[i+2]*EPSILON;
        }
    }
} 

int formatInferenceReqs(Image& idSDFMap, Image& stepMask, Matrix& points, Matrix& batch, const int imageW, const int imageH, dim3 gridSize, dim3 blockSize) {
    typedef thrust::device_ptr<uint> MatImgPtr;
    
    // d_mask == 1 if step, 6 if normal. required.
    MatImgPtr lastVal = thrust::exclusive_scan(
        thrust::device,
        (MatImgPtr)stepMask.deviceData.get(), 
        (MatImgPtr)(stepMask.deviceData.get() + stepMask.size()), 
        (MatImgPtr)idSDFMap.deviceData.get(),
        0
    );

    thrust::host_vector<uint> t(lastVal-1,lastVal);
    int batchSize = t[0];
    batch.shape.y = batchSize;

    // prepare batch
    createBatch<<<gridSize, blockSize>>> (
        batch.deviceData.get(),
        idSDFMap.deviceData.get(), 
        stepMask.deviceData.get(), 
        points.deviceData.get(), 
        imageW,
        imageH
    );

    //printCrap(idSDFMap,stepMask, points, batch);

    
    return batchSize;
}


Matrix points;
Matrix batch;
Matrix ray;
Matrix far;
Image stepMask;
Image idSDFMap;

int prevImageSize = -1;

void allocateBuffers(const int imageW, const int imageH) {
    int imageSize = imageW*imageH;

    points = Matrix(Shape(3, imageSize));
    batch = Matrix(Shape(3*6, imageSize));
    ray = Matrix(Shape(3, imageSize));
    far = Matrix(Shape(1, imageSize));
    stepMask = Image(Shape(imageH, imageW));
    idSDFMap = Image(Shape(imageH, imageW));

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
    int batchSize = formatInferenceReqs(
        idSDFMap,
        stepMask,
        points,
        batch,
        imageW,
        imageH,
        gridSize,
        blockSize
    );

    // march all rays simultaneossly. (so we can utilize batched gemm optimizations)
    for (int i = 0; i < MAX_STEPS; i ++) {   
        
        if (batchSize == 0) {
            //nothing to do!
            break;
        }

        // infer all points required
        sdf = nn.forward(batch, imageSize*6);  //the x6 is too ensure memory exists if entire image needs to be colored.

        // take step, updating mask, points, and ray position (tfar)
        singleMarch<<<gridSize, blockSize>>>(
            d_output,
            (const uint *)idSDFMap.deviceData.get(), 
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
        batchSize = formatInferenceReqs(
            idSDFMap,
            stepMask,
            points,
            batch,
            imageW,
            imageH,
            gridSize,
            blockSize
        );
        
    }
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
