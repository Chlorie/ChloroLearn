#pragma once

#include "../operand.h"

namespace chloro::operators
{
    Operand relu(Operand operand); // ReLU - Rectified linear unit
    Operand sigmoid(Operand operand); // Sigmoid function
    Operand softmax(Operand operand); // Softmax function
}
