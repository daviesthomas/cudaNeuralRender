
#include <Eigen/Eigen>

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

// cwd fetch
#ifdef WINDOWS
    #include <direct.h>
    #define GetCurrentDir _getcwd
#else
    #include <unistd.h>
    #define GetCurrentDir getcwd
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

#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"
#include "neuralUtils/image.hh"

typedef unsigned int uint;

// configuration options (can be set by arguments)
std::string neuralGeometryPath;
std::string renderSavePath;
bool singleImage;
std::string matcapPath;
uint width, height;

// global toggle for saving frame
bool doSaveNextFrame = false;

//////////////////////////////////////////////

dim3 blockSize(8, 8);
dim3 gridSize;

float3 viewRotation = make_float3(0.0, 180.0,0.0);
float3 viewTranslation = make_float3(.0, 0.0, -2.0f);

Eigen::Matrix4f normalMatrix;

// TODO: convert to eigen!
Eigen::MatrixXf transposedModelView(3,4);

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

Image matcap;
NeuralNetwork nn;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void render_kernel(
    dim3 gridSize, 
    dim3 blockSize, 
    uint *d_output, 
    uint imageW, 
    uint imageH, 
    NeuralNetwork nn,
    Image matcap
);

extern "C" void copyViewMatrices(float *invViewMatrix, size_t sizeofInvModelViewMat, float *normalMatrix, size_t sizeofNormalMatrix);

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


// render image using CUDA
void render()
{
    // copy invViewMatrix to constant memory.
    copyViewMatrices(transposedModelView.data(), sizeof(float4)*3, normalMatrix.data(), sizeof(float4)*4);


    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    
    render_kernel(
        gridSize, 
        blockSize, 
        d_output, 
        width, 
        height, 
        nn,
        matcap
    );

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    if (doSaveNextFrame) {
        unsigned char *h_output = (unsigned char *)malloc(width*height*4);
        checkCudaErrors(cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost));

        sdkSavePPM4ub(renderSavePath.c_str(), h_output, width, height);
        free(h_output);
        std::cout << "saved frame...\n";
        doSaveNextFrame = false;

        if (singleImage) {
            exit(0);
        }
    }
}

void updateViewMatrices() {
    Eigen::Affine3f modelView = Eigen::Affine3f::Identity();

    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-viewRotation.x*M_PI/180, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(-viewRotation.y*M_PI/180,  Eigen::Vector3f::UnitY());

    modelView.rotate(m);
    modelView.translate(Eigen::Vector3f(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z));

    normalMatrix = modelView.matrix().transpose().inverse();
    
    transposedModelView.row(0) = modelView.matrix().col(0);
    transposedModelView.row(1) = modelView.matrix().col(1);
    transposedModelView.row(2) = modelView.matrix().col(2);
}


// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    updateViewMatrices();

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

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
    const int SPACE = 32;
    switch (key)
    {   
        case SPACE:
            //save image here! 
            doSaveNextFrame = true;
            break;
        default:
            printf("you pressed a key! %d\n", key);
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

void generateSingleImage()
{   
    uint *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, width*height*sizeof(uint)));
    checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(uint)));

    updateViewMatrices();

    // call CUDA kernel, writing results to PBO
    copyViewMatrices(transposedModelView.data(), sizeof(float4)*3, normalMatrix.data(), sizeof(float4)*4);

    // Start timer 0 and process n loops on the GPU
    int nIter = 1;

    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    render_kernel(
        gridSize, 
        blockSize, 
        d_output, 
        width, 
        height, 
        nn,
        matcap
    );
    
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Error: render_kernel() execution FAILED");
    sdkStopTimer(&timer);

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = sdkGetTimerValue(&timer)/(nIter * 1000.0);
    printf("volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y);


    unsigned char *h_output = (unsigned char *)malloc(width*height*4);
    checkCudaErrors(cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost));

    sdkSavePPM4ub(renderSavePath.c_str(), h_output, width, height);

    cudaFree(d_output);

    free(h_output);

    cleanup();
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


void startRendering(int argc, char** argv) {
    initGL(&argc, argv);

    findCudaDevice(argc, (const char **)argv);

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

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
  {
      return *itr;
  }
  return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
  return std::find(begin, end, option) != end;
}

void usage() {
    std::cout << "Usage: neuralSDFRenderer [OPTION]... -i SOURCE.h5\n";
    std::cout << "         start interactive rendering of neuralGeometry file with default settings\n";
    std::cout << "   or: neuralSDFRenderer [OPTION]... -i SOURCE.h5 -o DEST.ppm\n";
    std::cout << "         render single image of neuralGeometry file and save to destination.ppm file\n";
    std::cout << "Options\n";
    std::cout << "\t-i input neuralGeometry path (string) default: neuralGeometries/armadillo.h5\n";
    std::cout << "\t-o output image path (string) default: volume.ppm\n";
    std::cout << "\t-H imageH (int)\n";
    std::cout << "\t-W imageW (int)\n";
    std::cout << "\t-M matcap file path (string) ( default: matcaps/blue.png )\n";
    std::cout << "\t-rx rotation in degree about x axis \n";
    std::cout << "\t-ry rotation in degree about y axis \n";
    std::cout << "\t-rz rotation in degree about z axis \n";
    std::cout << "\t--single if present, only a single frame is rendered and saved. (default: false)\n";
    std::cout << "\t-h (--help)\n";
}

void parseCmdOptions(int argc, char** argv)
{
    if (cmdOptionExists(argv, argv + argc, "-h")) {
        usage();
        exit(0);
    }
    if (cmdOptionExists(argv, argv+argc, "-i")){
        neuralGeometryPath = getCmdOption(argv, argv+argc, "-i");
    } else {
        neuralGeometryPath = "neuralGeometries/armadillo.h5";
    }
    if (cmdOptionExists(argv, argv+argc, "-o")){
        renderSavePath = atoi(getCmdOption(argv, argv+argc, "-o"));
    } else{
        renderSavePath = neuralGeometryPath + std::string(".ppm");
    }
    if (cmdOptionExists(argv, argv+argc, "-H")){
        height = atoi(getCmdOption(argv, argv+argc, "-H"));
    } else {
        height = 512;
    }
    if (cmdOptionExists(argv, argv+argc, "-W")){
        width = atoi(getCmdOption(argv, argv+argc, "-W"));
    } else {
        width = 512;
    }
    if (cmdOptionExists(argv, argv+argc, "-M")){
        matcapPath = getCmdOption(argv, argv+argc, "-M");
    } else {
        matcapPath = "matcaps/blue.png";
    }
    float rx, ry, z;
    if (cmdOptionExists(argv, argv+argc, "-rx")){
        rx = atof(getCmdOption(argv, argv+argc, "-rx"));
    } else {
        rx = 0.0;
    }
    if (cmdOptionExists(argv, argv+argc, "-ry")){
        ry = atof(getCmdOption(argv, argv+argc, "-ry"));
    } else {
        ry = 0.0;
    }

    viewRotation.x = rx;
    viewRotation.y = ry;
    
    if (cmdOptionExists(argv, argv+argc, "--single")) {
        singleImage = true;
    }
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

    parseCmdOptions(argc, argv);

    bool ok = nn.load(neuralGeometryPath);    
    if (!ok) {
        printf("Failed to initialize model (%s)... exiting \n", neuralGeometryPath.c_str());
        return 0;
    }   
    printf("Model initialized...\n\n");

    ok = matcap.loadPNG(matcapPath);
    if (!ok) {
        printf("Failed to load matcap file (%s)... exiting \n", matcapPath.c_str());
        return 0;
    }

    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    sdkCreateTimer(&timer);
    if (singleImage) {
        generateSingleImage();
    }
    else {
        startRendering(argc,argv);
    }
}




