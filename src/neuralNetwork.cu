#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"

/*
NeuralNetwork nn;

loadModelFromH5("model.h5", nn);

int BATCH_SIZE = 1;
Matrix Y;
Matrix X = Matrix(Shape(BATCH_SIZE,3));
X.allocateMemory();
X[0] = static_cast<float>(0.0);
X[1] = static_cast<float>(0.0);
X[2] = static_cast<float>(0.0);

X.copyHostToDevice();

Y = nn.forward(X);

Y.copyDeviceToHost();

printf("\nY:(%d,%d)\n", Y.shape.x, Y.shape.y);

for (int i=0; i< Y.shape.y; i++) {
    printf("%f\n",tanh(Y[i]));
}
*/

NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork() {
    for (auto layer: layers) {
        delete layer;
    }
}

void NeuralNetwork::addLayer(Layer* layer) {
    this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
    Matrix Z = X;

    for (auto layer: layers) {
        Z = layer->forward(Z);
    }

    Y = Z;
    return Y;
}

std::vector<Layer*> NeuralNetwork::getLayers() const {
	return layers;
}

int NeuralNetwork::getNumWeightParams() const {
    int numWeightParams = 0;
    for (auto layer: layers) {
        numWeightParams += layer->getNumWeightParams();
    }
    return numWeightParams;
}

int NeuralNetwork::getNumBiasParams() const {
    int numBiasParams = 0;
    for (auto layer: layers) {
        numBiasParams += layer->getNumBiasParams();
    }
    return numBiasParams;
}
