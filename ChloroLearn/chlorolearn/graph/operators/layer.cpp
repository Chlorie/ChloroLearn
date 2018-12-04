#include "layer.h"
#include "basic_operators.h"
#include "neural_network.h"

// ReSharper disable CppInconsistentNaming

namespace chloro::layers
{
    NodeRef dense_layer(Graph& graph, const NodeRef input, const size_t output_rows, Activation&& activation)
    {
        using namespace operators;
        const ArrayShape& shape = input.get().shape();
        if (shape.size() != 2 || shape[1] != 1)
            throw IllegalArgumentException("Input should be a column vector");
        const size_t row = shape[0];
        const NodeRef weights = graph.add_variable({ output_rows, row });
        const NodeRef bias = graph.add_variable({ output_rows, 1 });
        const NodeRef no_activation = graph.add_operator(matrix_multiply(weights, input) + bias);
        if (activation != nullptr) return graph.add_operator(activation(no_activation));
        return no_activation;
    }

    NodeRef convolutional_2d(Graph& graph, const NodeRef input, const size_t kernel_size,
        const size_t filter_amount, const size_t stride)
    {
        return convolutional_2d(graph, input,
            { kernel_size, kernel_size }, filter_amount, stride);
    }

    NodeRef convolutional_2d(Graph& graph, const NodeRef input, const ArrayShape& kernel_size,
        const size_t filter_amount, const size_t stride)
    {
        using namespace operators;
        if (kernel_size.size() != 2) throw IllegalArgumentException("Kernels should be 2D");
        const ArrayShape& shape = input.get().shape();
        if (shape.size() != 3)
            throw IllegalArgumentException("Input should be 3D (2D feature maps)");
        const NodeRef kernel_node = graph.add_variable({ filter_amount, kernel_size[0], kernel_size[1], shape[2] });
        return graph.add_operator(convolution_2d_with_padding(input, kernel_node, stride));
    }
}
