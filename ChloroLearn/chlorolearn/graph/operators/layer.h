#pragma once

#include <functional>

#include "../graph.h"

// ReSharper disable CppInconsistentNaming

/**
 * \brief Helper functions for adding neural network layers to a graph.
 */
namespace chloro::layers
{
    using Activation = std::function<Operand(Operand)>;

    /**
     * \brief Add a dense layer to the given graph.
     * \param graph The graph to add a layer into.
     * \param input A column vector as the input.
     * \param output_rows Specifies the dimension of the output vector.
     * \param activation Specifies which activation function is used, defaults to nullptr (no activation).
     * \return A node reference to the final output (after the activation) of the added layer.
     */
    NodeRef dense_layer(Graph& graph, NodeRef input, size_t output_rows, Activation&& activation = nullptr);

    NodeRef convolutional_2d(Graph& graph, NodeRef input, size_t kernel_size, size_t filter_amount, size_t stride = 1);
    NodeRef convolutional_2d(Graph& graph, NodeRef input, const ArrayShape& kernel_size,
        size_t filter_amount, size_t stride = 1);
}
