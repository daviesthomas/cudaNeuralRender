#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"

#include <iostream>
#include <time.h>
#include <math.h>


bool loadModelFromH5 (std::string fp, NeuralNetwork& nn, bool hostOnly) {
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
            hostOnly            // only allocate on host!
        ));
        layerCount ++;
    }
    return true;
}

int main () {
    NeuralNetwork nn;

    bool ok = loadModelFromH5("model.h5", nn, false);

    int BATCH_SIZE = 1;
    Matrix Y;
    Matrix X = Matrix(Shape(1,3));

    X.allocateMemory();

    X[0] = static_cast<float>(0.0);
    X[1] = static_cast<float>(0.0);
    X[2] = static_cast<float>(0.0);

    X.copyHostToDevice();

    Y = nn.forward(X);

    Y.copyDeviceToHost();

    for (int i=0; i< Y.shape.y; i++) {
        printf("%f %f\n",Y[i], tanh(Y[i]));
    }
}

