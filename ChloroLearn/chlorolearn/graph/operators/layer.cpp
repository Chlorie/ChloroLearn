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
        graph.set_variable(weights, Array<double>::random({ output_rows, row }, 0.0, std::sqrt(2.0 / (row + output_rows))));
        const NodeRef bias = graph.add_variable({ output_rows, 1 });
        const NodeRef no_activation = graph.add_operator(matrix_multiply(weights, input) + bias);
        if (activation != nullptr) return graph.add_operator(activation(no_activation));
        return no_activation;
    }

    NodeRef convolutional_2d(Graph& graph, const NodeRef input, const size_t kernel_size,
        const size_t filter_amount, const size_t stride, Activation&& activation)
    {
        return convolutional_2d(graph, input, { kernel_size, kernel_size },
            filter_amount, { stride, stride }, std::move(activation));
    }

    NodeRef convolutional_2d(Graph& graph, const NodeRef input, const ArrayShape& kernel_size,
        const size_t filter_amount, const ArrayShape& stride, Activation&& activation)
    {
        using namespace operators;
        if (kernel_size.size() != 2) throw IllegalArgumentException("Kernels should be 2D");
        const ArrayShape& shape = input.get().shape();
        if (shape.size() != 3)
            throw IllegalArgumentException("Input should be 3D (2D feature maps)");
        const ArrayShape kernel_shape{ filter_amount, kernel_size[0], kernel_size[1], shape[2] };
        const NodeRef kernel_node = graph.add_variable(kernel_shape);
        const double variance = 2.0 / (kernel_size[0] * kernel_size[1] * shape[2]);
        graph.set_variable(kernel_node, Array<double>::random(kernel_shape, 0.0, std::sqrt(variance)));
        const NodeRef no_activation = graph.add_operator(convolution_2d_with_padding(input, kernel_node, stride));
        if (activation != nullptr) return graph.add_operator(activation(no_activation));
        return no_activation;
    }
}
