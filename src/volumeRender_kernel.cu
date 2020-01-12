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
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

typedef unsigned int  uint;
typedef unsigned char uchar;


typedef struct
{
    float4 m[3];
} float3x4;

typedef struct
{
    float4 m[4];
} float4x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
__constant__ float4x4 c_normalMatrix;
__constant__ int c_coloringType = 0;  //default to ratio coloring.

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

struct Sphere
{
    float3 c;   //center
    float r;    //radius
};

const uint BACKGROUND_COLOR = 0;
const int COLOR_MASK_VAL = 6;
const float EPSILON = 0.001;
const int MAX_STEPS = 70;


// intersect ray with a sphere
__device__
bool intersectSphere(Ray ray, Sphere sphere, float *tnear, float *tfar)
{
    float3 Q = ray.o - sphere.c;
    float a = dot(ray.d, ray.d);
    float b = 2.0 * dot(Q, ray.d);
    float c = dot(Q,Q) - sphere.r*sphere.r;
    float discrim = b*b - 4*a*c;

    if (discrim > 0) {
        *tnear = (-b - sqrt(discrim)) / (2.0 * a); 
        *tfar =  (-b + sqrt(discrim)) / (2.0 *a);
        return true;
    }
    return false;
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

__device__
float4 mul(const float4x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = dot(v, M.m[3]);
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

    // With more shapes, this would be created elsewhere and passed in
    Sphere boundingSphere;
    boundingSphere.c = make_float3(0.0f);
    boundingSphere.r = 0.9;

    // find intersection with box
    float tnear, tfar;
    //int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    bool hit = intersectSphere(eyeRay, boundingSphere, &tnear, &tfar);

    // no need to march if ray never hits bounding primitive
    if ( !hit ) {
        d_mask[id] = 0;
        d_output[id] = BACKGROUND_COLOR;
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

__device__ 
float3 surfaceNormal(int idx, const float* d_sdf) {
    float3 normal;
    normal.x = d_sdf[idx] - d_sdf[idx+1];
    normal.y = d_sdf[idx+2] - d_sdf[idx + 3];
    normal.z = d_sdf[idx + 4] - d_sdf[idx + 5];
    normal = normalize(normal);
    return normal;
}

__device__
uint facingColor(float3 n, float3 rayDir) {
    float ratio = max(0.0, dot(n,-rayDir) );

    return rgbaFloatToInt(make_float4(ratio));
}


__device__ uint matCapColor(float3 normal, const uint* d_matcap, int matW, int matH) {
    float4 normal_eye4 = mul(c_normalMatrix, make_float4(normal.x, normal.y,normal.z, 0.0));
    float3 normal_eye = normalize(make_float3(normal_eye4.x, normal_eye4.y, normal_eye4.z));
    
    // TODO: matcap should be in texture memory... 
    int uvx = (normal_eye.x * 0.5 + 0.5) * matW;
    int uvy = (normal_eye.y * 0.5 + 0.5) * matH;
    
    int index = uvy * matW + uvx;

    if (index < 0) {
        return rgbaFloatToInt(make_float4(0.0f));
    }
    return d_matcap[index];
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
    int matcapW,
    int matcapH,
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
    const float3 ray = getFloat3(d_ray, 3*id);

    if (d_mask[id] >= COLOR_MASK_VAL) {
        // needs to be colored.
        float3 n = surfaceNormal(inferenceIndex, d_sdf);
        if (c_coloringType == 0) {
            d_output[id] = facingColor(n, ray);
        } else {
            d_output[id] = matCapColor(n, d_matcap, matcapW, matcapH);
        }
        
        d_mask[id] = 0;
        return;
    }

    float3 point = getFloat3(d_points, 3*id);

    const float tstep = tanh(d_sdf[inferenceIndex]);
    d_tfar[id] -= tstep;

    if (d_tfar[id] <= 0) {
        d_mask[id] = 0;
        d_output[id] = BACKGROUND_COLOR;
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

// simple function for debugging 
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
void createBatch (
    float* d_batch, 
    const uint* d_idSDFMap, 
    const uint* d_mask, 
    const float* d_points, 
    const int imageW, 
    const int imageH) 
{
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
    batch = Matrix(Shape(3*COLOR_MASK_VAL, imageSize)); 
    ray = Matrix(Shape(3, imageSize));
    far = Matrix(Shape(1, imageSize));
    stepMask = Image(Shape(imageW, imageH));
    idSDFMap = Image(Shape(imageW, imageH));

    // allocate them
    points.allocateMemory();
    ray.allocateMemory();
    far.allocateMemory();
    stepMask.allocateMemory();
    idSDFMap.allocateMemory();
    batch.allocateMemory();

    // we need to reinit the sdf buffer

}

extern "C"
void render_kernel(
    dim3 gridSize,
    dim3 blockSize,
    uint *d_output, 
    uint imageW, 
    uint imageH,
    NeuralNetwork& nn,
    Image matcap
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
        sdf = nn.forward(batch, imageSize*COLOR_MASK_VAL);  //the x6 is too ensure memory exists if entire image needs to be colored.
        // take step, updating mask, points, and ray position (tfar)
        singleMarch<<<gridSize, blockSize>>>(
            d_output,
            (const uint *)idSDFMap.deviceData.get(), 
            stepMask.deviceData.get(),
            (const float *)sdf.deviceData.get(), 
            points.deviceData.get(), 
            (const float *)ray.deviceData.get(),
            far.deviceData.get(),
            (const uint *)matcap.deviceData.get(),
            matcap.shape.x,
            matcap.shape.y,
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
    //TODO: set any ray that didnt converge to background color
    

}

extern "C"
void copyViewMatrices(float *invViewMatrix, size_t sizeofViewMatrix, float *normalMatrix, size_t sizeofNormalMatrix, int colorType)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofViewMatrix));
    checkCudaErrors(cudaMemcpyToSymbol(c_normalMatrix, normalMatrix, sizeofNormalMatrix));
    checkCudaErrors(cudaMemcpyToSymbol(c_coloringType, &colorType, sizeof(int)));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
