#pragma once

#include <chrono>

namespace chloro
{
    class Stopwatch final
    {
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
    private:
        TimePoint start_;
        TimePoint stop_;
    public:
        Stopwatch() :start_(Clock::now()) {}
        void stop() { stop_ = Clock::now(); }
        std::chrono::nanoseconds elapsed_time() const { return stop_ - start_; }
        double seconds() const { return elapsed_time().count() / 1e9; }
    };
}
