#pragma once

#include <vector>
#include <string>
#include "layers/layer.hh"

class NeuralNetwork {
    private:
        std::vector<Layer*> layers;
        
        Matrix Y;

    public:
        NeuralNetwork();
        ~NeuralNetwork();

        Matrix forward(Matrix X);
        void addLayer(Layer *layer);
        std::vector<Layer*> getLayers() const;
        bool load(std::string fp);
};