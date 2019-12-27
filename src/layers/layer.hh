#pragma once

#include <iostream>

#include "../neuralUtils/matrix.hh"

class Layer {
    protected:
        std::string name;

    public:
        virtual ~Layer() = 0;
        virtual Matrix& forward(Matrix& A) = 0;

        std::string getName() { return this->name; };
        //Shape getShape() { return this.shape; };
};

inline Layer::~Layer() {}