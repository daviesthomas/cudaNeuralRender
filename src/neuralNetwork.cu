#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"

NeuralNetwork::NeuralNetwork()
{}

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
        // we want this class to be ran from another kernel! 
        // TODO: have forward inference run entirely through __globals__
        Z = layer->forward(Z);
    }

    Y = Z;
    return Y;
}

std::vector<Layer*> NeuralNetwork::getLayers() const {
	return layers;
}
