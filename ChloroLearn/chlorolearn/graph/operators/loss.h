#pragma once

#include "../operand.h"

namespace chloro::operators
{
    /**
     * \brief Categorical cross entropy loss function. Usually used as the loss function for
     * classification tasks.
     * \param predicted The predicted value for the target one-hot vector. All components of
     * \a predicted should be positive, and sum up to 1. Usually \a predicted vector is an output
     * of \c softmax operation.
     * \param target A scalar value array (shape 1), containing a non-negative integer representing
     * the 0-based index of the value 1 in the target one-hot vector. Notice that the back-propagation
     * process will not proceed to this branch, that is, the \a target won't be updated (in case it's
     * an output of some operation).
     * \return A scalar value array containing the computed loss.
     */
    Operand categorical_cross_entropy(Operand predicted, Operand target);
}
