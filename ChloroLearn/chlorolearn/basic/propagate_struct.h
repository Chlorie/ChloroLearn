#pragma once

#include "array.h"

namespace chloro
{
    using InParam = const Array<double>&;
    using InParams = const std::vector<std::reference_wrapper<const Array<double>>>&;
    using OutParam = Array<double>;
    using OutParams = std::vector<OutParam>;
    using StateParam = Array<double>&;

    /** \brief An aggregate struct containing values used for forward propagating. */
    struct ForwardParams
    {
        InParams childs; /**< \brief Values of child nodes. */
        StateParam state; /**< \brief State of this node. */
        ForwardParams() = delete;
    };
    
    /** \brief An aggregate struct containing values used for back propagating. */
    struct BackwardParams
    {
        InParam gradient; /**< \brief Gradient passed from the parent node. */
        InParams childs; /**< \brief Values of child nodes. */
        InParam value; /**< \brief Current value of this node. */
        StateParam state; /**< \brief State of this node. */
        BackwardParams() = delete;
    };
}
