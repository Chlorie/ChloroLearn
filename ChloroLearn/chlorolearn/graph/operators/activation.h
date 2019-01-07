#pragma once

#include "../operand.h"

namespace chloro::operators
{
    /**
     * \brief ReLU (rectified linear unit) operation.
     * \param operand Input operand. Could be in any shape.
     * \return The output operand.
     */
    Operand relu(Operand operand);

    /**
     * \brief Leaky ReLU operation.
     * \param operand Input operand. Could be in any shape.
     * \return The output operand.
     */
    Operand leaky_relu(Operand operand);

    /**
     * \brief Sigmoid function.
     * \param operand Input operand. Could be in any shape.
     * \return The output operand.
     */
    Operand sigmoid(Operand operand);

    /**
     * \brief Softmax function.
     * \param operand Input operand. Could be in any shape.
     * \return The output operand.
     */
    Operand softmax(Operand operand);
}
