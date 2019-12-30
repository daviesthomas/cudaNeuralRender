#include "denseLayer.hh"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>

// KERNELS (kinda...)

//TODO: fix this! This will be faster than the naive implementation below... 
// for now I'll ignore and more forward with getting this class into volume renderer
cudaError_t denseLayerForward(
    float* W, float* A, float* Z, float* b, 
    int Wx, int Wy, int Ax, int Ay,
    int activation, float beta = 1.0, float alpha = 1.0) {

    // Aliases
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using ArchTag = cutlass::arch::Sm60;
    using OpClass = cutlass::arch::OpClassSimt;

    // Use Gemm defaults (for now...)
    using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<OpClass, //op class
                                                                    ArchTag, // arch tag
                                                                    float,  // element a
                                                                    float,  // element b
                                                                    float,  // element c
                                                                    float>; // element accum

    // epilogues run within gemm kernel, once matmul complete.
    // this feature is main reason we are using cutlass

    using activation_op = cutlass::epilogue::thread::LinearCombinationRelu<float,1>; // simt must operate on scalars!
    if (activation != ReLU) {
        using activation_op = cutlass::epilogue::thread::LinearCombination<float,1>;
    } 

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                ColumnMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                ColumnMajor,  // Layout of C matrix                                                             
                                                float,        // Element Accumulator Type
                                                OpClass,
                                                ArchTag,
                                                DefaultConfig::ThreadblockShape, 
                                                DefaultConfig::WarpShape, 
                                                DefaultConfig::InstructionShape,
                                                activation_op>;
                        
    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;
    
    // Construct the CUTLASS GEMM arguments object.
    CutlassGemm::Arguments args({Wx , Ay, Wy},  // Gemm Problem dimensions
                                {W, Wx},    // Tensor-ref for source matrix A
                                {A, Ax},    // Tensor-ref for source matrix B
                                {b, Ay},    // Tensor-ref for source matrix C
                                {Z, Ay},    // Tensor-ref for destination matrix D
                                {alpha, beta}); // Scalars used in the Epilogue


    // Launch the CUTLASS GEMM kernel.
    cutlass::cutStatus status = gemm_operator(args);

    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    if (status != cutlass::cutStatus::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}

__global__ void reluActivationForward(float* Z, float* A,
    int Z_x_dim, int Z_y_dim) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = fmaxf(Z[index], 0);
    }
}

__global__ void layerForward( float* W, float* A, float* Z, float* b,
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


// Dense Layer Class imp
DenseLayer::DenseLayer(
    std::string name, 
    std::vector<std::vector<float>> weights, 
    std::vector<float> biases,
    int activation, bool hostOnly
) 
{
    this->W = Matrix(Shape(weights.size(), weights[0].size()), hostOnly);
    this->numWeightParams = weights.size() * weights[0].size();
    this->b = Matrix(Shape(biases.size(),1), hostOnly);
    this->numBiasParams = biases.size();
    this->name = name;
    this->type = eDense;
    this->hostOnly = hostOnly;

    b.allocateMemory();
    W.allocateMemory();
    
    initializeBias(biases);
    initializeWeights(weights);

    this->activation = activation;
}

DenseLayer::~DenseLayer()
{ }

void DenseLayer::initializeBias(std::vector<float> biases) {
    
    for (int x = 0; x < biases.size(); x ++) {
        b[x] = biases[x];
    }

    if (!hostOnly) {
        b.copyHostToDevice();
    }

}

void DenseLayer::initializeWeights(std::vector<std::vector<float>> weights) {
    for (int x = 0; x < weights.size(); x++) {
        for (int y = 0; y < weights[0].size(); y ++) {
            W[y*W.shape.x + x] = weights[x][y];
        }
    }

    if (!hostOnly) {
        W.copyHostToDevice();
    }
}

Matrix& DenseLayer::forward(Matrix& A) {
    assert(W.shape.x == A.shape.y);

    this->A = A;
    Shape Z_shape(A.shape.x, W.shape.y);
    Z.maybeAllocateMemory(Z_shape);

    cudaError_t ok = computeAndStoreLayerOutput(A);

    checkCudaErrors(ok);

    return Z;   
}

cudaError_t DenseLayer::computeAndStoreLayerOutput(Matrix& A) {
    
    dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
                        (Z.shape.y + block_size.y - 1) / block_size.y);

    std::cout << W.shape.x << " " <<W.shape.y <<" " <<  A.shape.x <<" " << A.shape.y << std::endl;
                        
	layerForward<<<num_of_blocks, block_size>>>( W.deviceData.get(),
													   A.deviceData.get(),
													   Z.deviceData.get(),
													   b.deviceData.get(),
													   W.shape.x, W.shape.y,
                                                       A.shape.x, A.shape.y,
                                                       this->activation);

    Z.copyDeviceToHost();
    printf("%f \n", Z[0]);

    if (activation == ReLU) {
        dim3 bs(256);
        dim3 numBlocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

        reluActivationForward<<<numBlocks, bs>>>(Z.deviceData.get(), Z.deviceData.get(),Z.shape.x, Z.shape.y);
    }

    Z.copyDeviceToHost();
    printf("%f \n", Z[0]);

    return cudaSuccess;
}

int DenseLayer::getXDim() const {
	return W.shape.x;
}

int DenseLayer::getYDim() const {
	return W.shape.y;
}

Matrix DenseLayer::getWeightsMatrix() const {
	return W;
}

Matrix DenseLayer::getBiasVector() const {
	return b;
}

