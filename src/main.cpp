// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>

#include <iostream>
#include <time.h>
#include <math.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

const char *sSDKsample = "CUDA 3D Volume Render";

uint width = 64, height = 64;
dim3 blockSize(4, 4);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];


GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

int *pArgc;
char **pArgv;

NeuralNetwork nn;
Matrix NetworkWeights, NetworkBiases, NetworkDims;  // storage of network crap
Matrix A, Z;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void render_kernel(
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
    int numLayers);

extern "C" void sdf_kernel(
    float* weights, 
    float* biases, 
    float* dims,
    float* A,
    float* Z,
    int numLayers);

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer();

bool loadModelFromH5 (std::string fp, NeuralNetwork& nn) {
    HighFive::File file(fp, HighFive::File::ReadOnly);

    std::vector<std::string> kerasLayers = file.listObjectNames();
    int layerCount = 0;

    for (std::vector<std::string>::iterator it = kerasLayers.begin() ; it != kerasLayers.end(); ++it) {
        // for each layer, copy weights to eigen
        HighFive::ObjectType objType = file.getObjectType(*it);

        if (objType != HighFive::ObjectType::Group) {
            std::cout << "Unsupported Layer\n";
            return false;
        }

        HighFive::Group group = file.getGroup(*it);
        int n = group.getNumberObjects();
        
        if (n != 1) {
            std::cout << "Unsupported Layer\n";
            return false;
        }

        group = group.getGroup(*it);
        std::vector<std::string> matNames = group.listObjectNames();

        std::vector<std::vector<float>> weights;
        std::vector<float> biases;

        for (std::vector<std::string>::iterator matIt = matNames.begin(); matIt != matNames.end(); ++matIt) {
            objType = group.getObjectType(*matIt);
            if (objType != HighFive::ObjectType::Dataset) {
                std::cout << "Unsupported Layer\n";
                return false;
            }

            // parse the weights and biases
            HighFive::DataSet dataset = group.getDataSet(*matIt);
            std::vector<size_t> dim = dataset.getDimensions();

            if (dim.size() == 1) {
                dataset.read(biases);
            } else if (dim.size() == 2) {
                dataset.read(weights);
            }
            else {
                std::cout << "Unsupported layer, to many dims!\n";
                return false;
            }
        }

        int activation = ReLU; //RELU
        if  ((it != kerasLayers.end()) && (next(it) == kerasLayers.end())) {
            activation = Tanh; 
        }

        nn.addLayer(new DenseLayer(
            std::string("Dense_") + std::to_string(layerCount), 
            weights, 
            biases, 
            activation,     
            true            // only allocate on host!
        ));
        layerCount ++;
    }
    return true;
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Volume Render: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

// render image using CUDA
void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    
    render_kernel(
        gridSize, 
        blockSize, 
        d_output, 
        width, 
        height, 
        NetworkWeights.deviceData.get(), 
        NetworkBiases.deviceData.get(), 
        NetworkDims.deviceData.get(),
        A.deviceData.get(),
        Z.deviceData.get(),
        nn.getLayers().size()
    );
    
    

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {   
        default:
            printf("you pressed a key!\n");

    }

    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    width = w;
    height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    // removed for now, while we have no buffer to free
    //freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    if (!isGLVersionSupported(2,0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions are missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool initializeSDFNetwork(std::string modelPath) {
    //load weights to our NN class.
    loadModelFromH5(modelPath, nn);

    std::vector<Layer*> layers = nn.getLayers();

    int numWeightParams = nn.getNumWeightParams();
    int numBiasParams = nn.getNumBiasParams();

    // init two "flat" matrices
    //TODO: these should be allocated into constant memory if possible... maybe impossible due to mem limits (~64kb...)
    NetworkWeights = Matrix(Shape(numWeightParams, 1));     // store all network weights
    NetworkBiases = Matrix(Shape(numBiasParams, 1));        // stores all network biases
    NetworkDims = Matrix(Shape(layers.size()*3, 1));        // stores tuples (M,N,K) for matmul

    // sets the number of concurrent inferences. A and Z are allocated for each concurrent model.
    //TODO: unsure what this should actually be set too... im struggling to understand actual
    //          number of concurrent.
    // this is an insane waste of memory atm...
    int batchSize = height*width;

    std::cout << "BatchSize: " << batchSize << std::endl;
    
    A = Matrix(Shape(batchSize,32));
    Z = Matrix(Shape(batchSize,32));

    NetworkWeights.allocateMemory();
    NetworkBiases.allocateMemory();
    NetworkDims.allocateMemory();
    A.allocateMemory();
    Z.allocateMemory();

    int bPos = 0;
    int wPos = 0;
    int dPos = 0;

    for (std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); ++it) {
        if ((*it)->getType() != eDense) {
            std::cout << "Invalid layer type detected... exiting...\n";
            return 0;
        }

        DenseLayer* layer = static_cast<DenseLayer*>(*it);
        
        // copy layer weights to all weight matrix
        Matrix wData = layer->getWeightsMatrix();
        Shape wShape = layer->getWeightsMatrix().shape;
        Matrix bData = layer->getBiasVector();
        Shape bShape = layer->getBiasVector().shape;

        for (int i = 0; i < wShape.x*wShape.y; i ++) {
            NetworkWeights[wPos+i] = wData[i];
        }
        
        for (int i = 0; i < bShape.x*bShape.y; i ++) {
            NetworkBiases[bPos+i] = bData[i];
        }
        
        // M, N, K
        NetworkDims[dPos] = wShape.x;
        NetworkDims[dPos + 1] = bShape.y;
        NetworkDims[dPos + 2] = wShape.y;

        wPos += wShape.x * wShape.y;
        bPos += bShape.x * bShape.y;
        dPos += 3;
    }

    // copy our flat matrices to device memory.
    NetworkWeights.copyHostToDevice();
    NetworkBiases.copyHostToDevice();
    NetworkDims.copyHostToDevice();

    return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    bool ok = initializeSDFNetwork("model.h5");
    
    if (!ok) {
        printf("Failed to initialize model... exiting \n");
        return 0;
    }   
    printf("Model initialized...\n\n");

    //sdf_kernel(
     //   NetworkWeights.deviceData.get(), 
      //  NetworkBiases.deviceData.get(), 
       // NetworkDims.deviceData.get(),
        //A.deviceData.get(),
       // Z.deviceData.get(),
       // nn.getLayers().size()
    //);

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);

    findCudaDevice(argc, (const char **)argv);
    
    sdkCreateTimer(&timer);

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    glutMainLoop();
}



