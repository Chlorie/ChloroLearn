#include <iostream>

#include "chlorolearn/basic/array.h"
#include "chlorolearn/graph/graph.h"
#include "chlorolearn/graph/operators/basic_operators.h"
#include "chlorolearn/utility/stopwatch.h"

int main()
{
    using namespace chloro;
    using namespace operators;
    // Test: Several operators
    Graph graph;
    const NodeRef x = graph.add_constant({ {1,2},{3,4} });
    const NodeRef y = graph.add_constant({ {1,2},{3,4} });
    const NodeRef mul = graph.add_operator(matrix_multiply(x, y));
    Stopwatch stopwatch;
    const Array<double>& result = graph.get_value(mul);
    stopwatch.stop();
    std::cout << "Result = " << result << ", finished in " << stopwatch.elapsed_time().count() / 1e9 << "s\n";
    getchar();
    return 0;
}
