#pragma once

#include "../operand.h"

namespace chloro::operators
{
    // Cross entropy loss for classification use
    // Loss will not back propogate to the target branch
    Operand categorical_cross_entropy(Operand predicted, Operand target);
}
