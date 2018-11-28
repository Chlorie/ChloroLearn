#include <iostream>

#include "chlorolearn/basic/array.h"
#include "chlorolearn/graph/graph.h"
#include "chlorolearn/graph/operators/basic_operators.h"
#include "chlorolearn/graph/operators/activation_functions.h"
#include "chlorolearn/testing/stopwatch.h"

int main()
{
    using namespace chloro;
    using namespace operators;
    // Test: Several operators
    Graph graph;
    const NodeRef x = graph.add_variable({ 2,2 });
    const NodeRef c = graph.add_constant({ {1,3},{2,4} });
    const NodeRef target = graph.add_operator(sum(power(x - c, 2)));
    std::cout << "x = " << graph.get_value(x) << ", target = " << graph.get_value(target) << '\n';
    Stopwatch stopwatch;
    graph.optimize(1000000, target);
    stopwatch.stop();
    std::cout << "After 1000000 iterations (elapsed time " << stopwatch.elapsed_time().count() / 1e9 << "s)\n";
    std::cout << "x = " << graph.get_value(x) << ", target = " << graph.get_value(target) << '\n';
    getchar();
    return 0;
}
