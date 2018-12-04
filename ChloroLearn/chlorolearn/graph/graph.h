#pragma once

#include <string>
#include <list>
#include <initializer_list>

#include "node.h"
#include "input_param.h"
#include "input_pack.h"
#include "operand.h"

namespace chloro
{
    /**
     * \brief A class representing a flow graph. All computational works are done
     * through manipulations of a \c Graph. All the operations in a graph is lazy-evaluated,
     * that is, the values are calculated every time you call the \c get_value method,
     * but not when you construct the graph.
     */
    class Graph final
    {
    private:
        std::list<Node> nodes_;
        void input(Node& node, const Array<double>& value) const;
        static void update_dag(Node& node);
        void forward_propagate(Node& node, std::initializer_list<InputParam> input_params = {});
    public:
        /** \brief Constructs an empty graph. */
        Graph() = default;
        /**
         * \brief Add an \c Input node of a specific shape into this graph.
         * \param shape The shape of the added node. Defaults to { 1 } (scalar input).
         * \return A reference to the added node.
         */
        Node& add_input(const ArrayShape& shape = { 1 });
        /**
         * \brief Add a \c Variable node of a specific shape into this graph.
         * \param shape The shape of the added node. Defaults to { 1 } (scalar input).
         * \return A reference to the added node.
         */
        Node& add_variable(const ArrayShape& shape = { 1 });
        /**
         * \brief Add a \c Constant node containing a constant array into this graph.
         * \param array The constant value of the node.
         * \return A reference to the added node.
         */
        Node& add_constant(const Array<double>& array);
        /**
         * \brief Add a \c Constant node containing a constant array into this graph.
         * \param array A temporary array that is to be moved into the constant node.
         * \return A reference to the added node.
         */
        Node& add_constant(Array<double>&& array);
        /**
         * \brief Add an \c Operand (a series of <tt>Operator</tt>s, please see documentation for \c Operand for
         * more info) into this graph.
         * \param list The temporary \c Operand to add into the graph.
         * \return A reference to the added node.
         */
        Node& add_operator(Operand&& list);
        /**
         * \brief Evaluates a node in the graph. Evaluates the \c Operator nodes in evaluation mode, that is
         * not updating state variables and not performing computations like dropouts. For more information, see
         * \c Operator.
         * \param node The node to evaluate.
         * \param input_params An \c std::initializer_list of <tt>InputParam</tt>s for \c Input nodes.
         * Defaults to an empty list.
         * \return The result of the evaluation.
         */
        const Array<double>& get_value(Node& node, std::initializer_list<InputParam> input_params = {});
        /**
         * \brief Explicitly set the value of a \c Variable node. Can be used in order to customize graph
         * saving and loading.
         * \param node The \c Variable node to set.
         * \param value The value to set the node to.
         */
        void set_variable(Node& node, const Array<double>& value) const;
        /**
         * \brief Optimize the target once using gradient descent method.
         * \param target The target \c Operator node to minimize.
         * \param input_params An \c std::initializer_list of <tt>InputParam</tt>s for \c Input nodes.
         * \param learning_rate Learning rate (step size relative to the gradient). Defaults to 1e-3.
         */
        void optimize_once(Node& target, std::initializer_list<InputParam> input_params = {}, double learning_rate = 1e-3);
        /**
         * \brief Optimize the target several times using SGD (stochastic gradient descent) method.
         * \param batch_size How many times to perform the SGD method.
         * \param target The target \c Operator node to minimize.
         * \param input_pack An \c std::initializer_list of <tt>InputPack</tt>s for \c Input nodes.
         * \param learning_rate Learning rate (step size relative to the gradient). Defaults to 1e-3.
         */
        void optimize(size_t batch_size, Node& target, std::initializer_list<InputPack> input_pack = {},
            double learning_rate = 0.001);
        /**
         * \brief Save current values of variables in the graph to a data file.
         * \param path The full path or relative path to the data file.
         */
        void save_variables(const std::string& path) const;
        /**
         * \brief Load values of variables in the graph from a data file.
         * \param path The full path or relative path to the data file.
         */
        void load_variables(const std::string& path);
    };
}
