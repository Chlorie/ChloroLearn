#pragma once

#include <cmath>

#include "../operand.h"

namespace chloro
{
    Operand operator+(Operand left, Operand right); /**< \brief See \c operators::add. */
    Operand operator-(Operand left, Operand right); /**< \brief See \c operators::subtract. */
    Operand operator*(Operand left, Operand right); /**< \brief See \c operators::multiply. */
    Operand operator/(Operand left, Operand right); /**< \brief See \c operators::divide. */

    /**
     * \brief This namespace contains most functions that work like operators, that is,
     * taking some operands as input and returning an operand as a result. For more
     * information on the actual \c Operand class, please refer to the corresponding
     * header and documentation.
     */
    namespace operators
    {
        /**
         * \brief Identity function.
         * \param operand The input operand.
         * \return A result operand that evaluates to the same value as the input.
         */
        Operand identity(Operand operand);

        /**
         * \brief Add operator. Also in the form of \c operator+.
         * \return The result operand the evaluates to \a left+right.
         */
        Operand add(Operand left, Operand right);
        /**
         * \brief Subtract operator. Also in the form of \c operator-.
         * \return The result operand the evaluates to \a left-right.
         */
        Operand subtract(Operand left, Operand right);
        /**
         * \brief Multiply operator. Also in the form of \c operator*.
         * \return The result operand the evaluates to \a left*right. Notice that the 
         * multiplication is element-wise. For matrix multiplication, use the function
         * \c matrix_multiply.
         */
        Operand multiply(Operand left, Operand right);
        /**
         * \brief Divide operator. Also in the form of \c operator/.
         * \return The result operand the evaluates to \a left/right.
         */
        Operand divide(Operand left, Operand right);
        /**
         * \brief Matrix multiplication operator.
         * \param left Left operand, should be a 2D array (a matrix).
         * \param right Right operand, should also be a 2D array (a matrix).
         * \return The result operand evaluates to \a left matrix multiplies \a right.
         */
        Operand matrix_multiply(Operand left, Operand right);
        /**
         * \brief Dot product of two vectors. The operands doesn't need to be real "vectors".
         * This operator is a short-hard for multiplying \a left and \a right and doing a
         * \c sum operation on the result.
         * \return The rersult evaluates to the dot product of the two arrays.
         */
        Operand dot(Operand left, Operand right);

        /**
         * \brief Outputs an array with the given shape filled with a specific scalar value.
         * \param scalar A scalar valued array (shape of 1) that needs to be repeated.
         * \param shape The shape of the result operand.
         * \return The result evaluates to the same result as \c Array<double>::repeat(scalar[0],
         * \a shape).
         */
        Operand repeat(Operand scalar, const ArrayShape& shape);
        /**
         * \brief Reshape an operand.
         * \param input The operand containing the array that needs to be reshaped.
         * \param shape The shape of the result operand.
         * \return The result evaluates to an array containing the same values as the
         * input, but reshaped to the \a shape.
         */
        Operand reshape(Operand input, const DefaultableArrayShape& shape);
        /**
         * \brief Calculates the element-wise sum of the operand.
         * \return A scalar shaped operand with the sum.
         */
        Operand sum(Operand operand);

        /**
         * \brief Raise the \a base to the power \a exponent, with a constant
         * \a exponent specified.
         * \return Evaluates to the array resulted from performing an element-wise
         * power operation on the input operand \a base.
         */
        Operand power(Operand base, double exponent);
        /**
         * \brief Raise the \a base to the power \a exponent, with a constant
         * \a base specified. The \a base constant defaults to the natural base e.
         * \return Evaluates to the array resulted from performing an element-wise
         * power operation on the input operand \a base.
         */
        Operand exp(Operand exponent, double base = std::exp(1.0));
    }
}
