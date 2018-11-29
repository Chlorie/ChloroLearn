#pragma once

#include "../operand.h"

namespace chloro::operators
{
    Operand relu(Operand operand); // ReLU - Rectified linear unit
    Operand softmax(Operand operand); // Softmax function
}
