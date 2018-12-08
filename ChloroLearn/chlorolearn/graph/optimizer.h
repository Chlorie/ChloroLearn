#pragma once

#include <functional>

#include "../basic/array.h"

namespace chloro
{
    using Optimizer = std::function<Array<double>(const Array<double>&)>;
    using OptimizerGenerator = std::function<Optimizer()>;

    /** \brief Provide some common optimizers like SGD and Adam. */
    namespace optimizers
    {
        /**
         * \brief Stochastic gradient descent optimizer.
         * \param rate Learning rate.
         * \return The result optimizer.
         */
        Optimizer sgd(double rate = 0.001);

        /**
         * \brief Adam (Adaptive moment estimation) optimizer.
         * \param alpha Step size, or learning rate.
         * \param beta_1 Exponential decay rate for first moment estimates.
         * \param beta_2 Exponential decay rate for second moment estimates.
         * \param epsilon A small positive value to avoid division by zero.
         * \return The result optimizer.
         */
        Optimizer adam(double alpha = 0.001, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8);
    }
}
