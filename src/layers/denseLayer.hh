#pragma once

#include "layer.hh"
#include <vector>

enum Activation {
    ReLU,
    Tanh    // we don't support tanh hidden... only final...
};

class DenseLayer : public Layer {
    private:
        Matrix W;
        Matrix b;
        Matrix A;
        Matrix Z;

        int activation;

        void initializeWeights(std::vector<std::vector<float>> weights);
        void initializeBias(std::vector<float> biases);

        cudaError_t computeAndStoreLayerOutput(Matrix& A);

    public:
        DenseLayer(
            std::string name, 
            std::vector<std::vector<float>> weights, 
            std::vector<float> biases,
            int activation
        );

        ~DenseLayer();

        Matrix &forward(Matrix& A);
        
        int getXDim() const;
        int getYDim() const;
        Matrix getWeightsMatrix() const;
        Matrix getBiasVector() const;
};