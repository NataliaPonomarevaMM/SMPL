#include <iostream>
#include <cstdlib>
#include <chrono>
#include <gtest/gtest.h>
#include "smpl/smpl.h"

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

int main(int argc, char const *argv[]) {
//    ::testing::InitGoogleTest(&argc, const_cast<char **>(argv));
//    return RUN_ALL_TESTS();

    smpl::SMPL model;
    auto begin = clk::now();
    model.init("/home/nponomareva/DoubleFusion/data/smpl_female.json");
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end - begin);
    std::cout << "Time duration to load SMPL: " << (double)duration.count() / 1000 << " s" << std::endl;

    float theta[72] = {0}, beta[10] = {0};

    for (int i = 0; i < 72; i++)
        theta[i] = rand();
    for (int i = 0; i < 72; i++)
        beta[i] = rand();

    const int64_t LOOPS = 100;
    duration = std::chrono::duration_cast<ms>(end - end);
    for (int64_t i = 0; i < LOOPS; i++) {
        begin = clk::now();
        model.lbs_for_model(beta, theta);
        end = clk::now();
        auto time = std::chrono::duration_cast<ms>(end - begin);
        std::cout << i << ") SMPL: " << (double)time.count() << " ms" << std::endl;
        duration += time;
    }
    std::cout << "Time duration to run SMPL: " << (double)duration.count() / LOOPS << " ms" << std::endl;

    return 0;
}