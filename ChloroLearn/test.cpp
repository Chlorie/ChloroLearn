#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "chlorolearn/basic/array.h"
#include "chlorolearn/graph/graph.h"
#include "chlorolearn/utility/stopwatch.h"
#include "chlorolearn/utility/utility.h"
#include "chlorolearn/graph/operators/basic_operators.h"
#include "chlorolearn/graph/operators/activation.h"
#include "chlorolearn/graph/operators/loss.h"

using namespace chloro;
using namespace operators;
using DataValues = std::vector<Array<double>>;

void load_training_data(const std::string& file_name, DataValues& x, DataValues& y)
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
        y.push_back(Array<double>{ value });
        x.push_back(Array<double>::zeros({ 784,1 }));
        Array<double>& array = x.back();
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
    const NodeRef input = graph.add_input({ 784,1 });
    const NodeRef weight_1 = graph.add_variable({ 100,784 });
    const NodeRef bias_1 = graph.add_variable({ 100,1 });
    const NodeRef output_1 = graph.add_operator(relu(matrix_multiply(weight_1, input) + bias_1));
    const NodeRef weight_2 = graph.add_variable({ 30,100 });
    const NodeRef bias_2 = graph.add_variable({ 30,1 });
    const NodeRef output_2 = graph.add_operator(relu(matrix_multiply(weight_2, output_1) + bias_2));
    const NodeRef weight_3 = graph.add_variable({ 10,30 });
    const NodeRef bias_3 = graph.add_variable({ 10,1 });
    const NodeRef predicted = graph.add_operator(softmax(sigmoid(matrix_multiply(weight_3, output_2) + bias_3)));
    const NodeRef target = graph.add_input();
    const NodeRef loss = graph.add_operator(categorical_cross_entropy(predicted, target));

    // Perform the optimization
    size_t counter = 0;
    const size_t batch_size = 10000;
    bool not_update_dag = false;

    graph.load_variables("result_300000.var");

    while (true)
    {
        // Optimize
        {
            Stopwatch stopwatch;
            graph.optimize(batch_size, loss, { {input, train_x}, {target, train_y} }, 1e-3, not_update_dag);
            stopwatch.stop();
            counter += batch_size;
            std::cout << counter << " times of back propagation finished -- "
                << "Last " << batch_size << " elapsed " << stopwatch.seconds() << "s\n";
        }
        // Evaluation
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
        }
        // Save weights and biases
        if (counter % 50000 == 0) graph.save_variables("result_" + std::to_string(counter) + ".var");
        not_update_dag = true;
    }
}
