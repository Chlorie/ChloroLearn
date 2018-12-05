#pragma once

#include <functional>

#include "../graph.h"

// ReSharper disable CppInconsistentNaming

/**
 * \brief Helper functions for adding neural network layers to a graph.
 * \details A neural network layer often consists of multiple nodes, for example, a dense layer
 * (fully connected layer) consists of two variable nodes for weights and bias, two operator nodes
 * for matrix multiplication and addition, and possibly another operator node for the activation
 * function. These functions are to simplify the process of adding these nodes to a graph.
 */
namespace chloro::layers
{
    using Activation = std::function<Operand(Operand)>; /**< Activation function (operator). */

    /**
     * \brief Add a dense layer to the given graph.
     * \param graph The graph to add a layer into.
     * \param input A column vector as the input.
     * \param output_rows Specifies the dimension of the output vector.
     * \param activation Specifies which activation function is used, defaults to nullptr (no activation).
     * \return A node reference to the final output (after the activation) of the added layer.
     */
    NodeRef dense_layer(Graph& graph, NodeRef input, size_t output_rows, Activation&& activation = nullptr);

    /**
     * \brief Add a 2D convolutional layer to the given graph.
     * \param graph The graph to add a layer into
     * \param input A 3D array of shape (rows x columns x channels) as the input.
     * \param kernel_size Side length of the kernel.
     * \param filter_amount The amount of filters.
     * \param stride Convolution stride, same for the two dimensions.
     * \param activation Specifies which activation function is used, defaults to nullptr (no activation).
     * \return A node reference to the final output of the added layer.
     */
    NodeRef convolutional_2d(Graph& graph, NodeRef input, size_t kernel_size, size_t filter_amount,
        size_t stride = 1, Activation&& activation = nullptr);
    /**
     * \brief Add a 2D convolutional layer to the given graph.
     * \param graph The graph to add a layer into
     * \param input A 3D array of shape (rows x columns x channels) as the input.
     * \param kernel_size Shape of the kernel, should be a 2D array shape.
     * \param filter_amount The amount of filters.
     * \param stride Convolution stride, should be a 2D array shape.
     * \param activation Specifies which activation function is used, defaults to nullptr (no activation).
     * \return A node reference to the final output of the added layer.
     */
    NodeRef convolutional_2d(Graph& graph, NodeRef input, const ArrayShape& kernel_size,
        size_t filter_amount, const ArrayShape& stride = { 1, 1 }, Activation&& activation = nullptr);
}
