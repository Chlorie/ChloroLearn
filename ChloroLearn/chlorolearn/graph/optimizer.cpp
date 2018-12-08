#include "optimizer.h"

namespace chloro::optimizers
{
    Optimizer sgd(const double rate) { return [=](const Array<double>& gradient) { return rate * gradient; }; }

    Optimizer adam(const double alpha, const double beta_1, const double beta_2, const double epsilon)
    {
        class Functor
        {
        private:
            const double alpha_;
            const double beta_1_;
            const double beta_2_;
            const double epsilon_;
            double beta_1_t_ = 1.0;
            double beta_2_t_ = 1.0;
            Array<double> first_;
            Array<double> second_;
            bool first_evaluation_ = true;
        public:
            Functor(const double alpha, const double beta_1, const double beta_2, const double epsilon)
                :alpha_(alpha), beta_1_(beta_1), beta_2_(beta_2), epsilon_(epsilon) {}
            Array<double> operator() (const Array<double>& gradient)
            {
                if (first_evaluation_)
                {
                    const ArrayShape& shape = gradient.shape();
                    first_ = Array<double>::zeros(shape);
                    second_ = Array<double>::zeros(shape);
                    first_evaluation_ = false;
                }
                beta_1_t_ *= beta_1_;
                beta_2_t_ *= beta_2_;
                first_ = beta_1_ * first_ + (1 - beta_1_) * gradient;
                second_ = beta_2_ * second_ + (1 - beta_2_) * gradient * gradient;
                first_ /= 1 - beta_1_t_;
                second_ /= 1 - beta_2_t_;
                return alpha_ * first_ / (second_.apply([](const double value) { return std::sqrt(value); }) + epsilon_);
            }
        };
        return Functor(alpha, beta_1, beta_2, epsilon);
    }
}
