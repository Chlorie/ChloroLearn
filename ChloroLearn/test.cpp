#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "chlorolearn/basic/array.h"
#include "chlorolearn/graph/graph.h"
#include "chlorolearn/utility/stopwatch.h"
#include "chlorolearn/utility/utility.h"
#include "chlorolearn/graph/operators/activation.h"
#include "chlorolearn/graph/operators/layer.h"
#include "chlorolearn/graph/operators/loss.h"
#include "chlorolearn/graph/operators/neural_network.h"
#include "chlorolearn/graph/optimizer.h"

void load_training_data(const std::string& file_name, chloro::DataValues& x, chloro::DataValues& y)
{
    std::ifstream stream{ file_name };
    // Read the first row and discard it...
    {
        std::string temp;
        std::getline(stream, temp);
    }
    while (true)
    {
        double value;
        stream >> value;
        if (!stream.good()) break;
        y.push_back(chloro::Array<double>{ value });
        x.push_back(chloro::Array<double>::zeros({ 28,28,1 }));
        chloro::Array<double>& array = x.back();
        for (size_t i = 0; i < 784; i++)
        {
            char temp;
            stream >> temp >> value;
            array[i] = value / 255;
        }
        if (y.size() % 7000 == 0) std::cout << "Loaded " << y.size() << " data rows\n";
    }
    stream.close();
}

int main()
{
    using chloro::Graph;
    using chloro::NodeRef;
    using chloro::Stopwatch;
    using chloro::DataValues;
    namespace opr = chloro::operators;
    namespace lyr = chloro::layers;
    namespace opt = chloro::optimizers;

    // Test: Real MNIST data test
    DataValues x, y;
    DataValues train_x, train_y;
    DataValues test_x, test_y;

    // Load data
    std::cout << "Loading training data...\n";
    load_training_data("train.csv", x, y);
    std::cout << "Complete\n";

    // Split train and test set
    train_test_split(x, y, train_x, train_y, test_x, test_y);
    std::cout << "Splitted training and test set\n";

    // Construct the graph
    Graph graph;
    const NodeRef input = graph.add_input({ 28,28,1 });
    const NodeRef conv_1 = lyr::convolutional_2d(graph, input, 3, 16, 1, opr::relu);
    const NodeRef conv_2 = lyr::convolutional_2d(graph, conv_1, 3, 16, 1, opr::relu);
    const NodeRef pool_1 = graph.add_operator(opr::max_pool_2d(conv_2, 2));
    const NodeRef dropout_1 = graph.add_operator(opr::dropout(pool_1, 0.25));
    const NodeRef conv_3 = lyr::convolutional_2d(graph, dropout_1, 3, 32, 1, opr::relu);
    const NodeRef conv_4 = lyr::convolutional_2d(graph, conv_3, 3, 32, 1, opr::relu);
    const NodeRef pool_2 = graph.add_operator(opr::max_pool_2d(conv_4, 2));
    const NodeRef dropout_2 = graph.add_operator(opr::dropout(pool_2, 0.25));
    const NodeRef flat = graph.add_operator(opr::flatten(dropout_2));
    const NodeRef dense_1 = lyr::dense_layer(graph, flat, 256, opr::relu);
    const NodeRef dense_2 = lyr::dense_layer(graph, dense_1, 128, opr::relu);
    const NodeRef predicted = lyr::dense_layer(graph, dense_2, 10, opr::softmax);
    const NodeRef target = graph.add_input();
    const NodeRef loss = graph.add_operator(opr::categorical_cross_entropy(predicted, target));

    graph.randomize_variables(0.0, 0.1);

    const size_t batch_size = 1000;
    const size_t save_period = 10000;

    graph.load_variables("result_190000.var");

    // Start optimization
    graph.optimize(loss, { {input, train_x}, {target, train_y} }, opt::sgd(1e-3), batch_size,
        [&](const double time) // Batch callback
        {
            static size_t counter = 0;
            counter += batch_size;
            std::cout << counter << " passes finished -- Last iteration elapsed " << time << "s\n";
            if (counter % save_period == 0)
            {
                const size_t test_size = test_x.size();
                size_t correct_amount = 0;
                Stopwatch stopwatch;
                for (size_t i = 0; i < test_size; i++)
                {
                    const std::vector<double>& result = graph.get_value(predicted, { {input, test_x[i]} }).data();
                    const size_t max_element = std::distance(result.begin(), std::max_element(result.begin(), result.end()));
                    if (max_element == size_t(test_y[i][0])) correct_amount++;
                }
                stopwatch.stop();
                std::cout << "Evaluation finished, elapsed time " << stopwatch.seconds() << "s -- "
                    << "Correct / Total = " << correct_amount << " / " << test_size
                    << " = " << double(correct_amount) / test_size << '\n';
                graph.save_variables("result_" + std::to_string(counter) + ".var");
            }
        },
        [](const double time) // Epoch callback
        {
            static size_t counter = 0;
            counter++;
            std::cout << counter << " epochs finished -- Last epoch elapsed " << time << "s\n";
        });
}
