#pragma once

#include <vector>

#include "../basic/array.h"
#include "node.h"

namespace chloro
{
    /** \brief An aggregate struct for providing a series of inputs for an \c Input node at once. */
    struct InputPack final
    {
        Node& input; /**< Reference to the node containing an \c Input content. */
        const std::vector<Array<double>>& pack; /**< Const reference to the vector of input arrays. */
        InputPack() = delete;
    };
}
