#include <iostream>
#include <cstdlib>
#include <chrono>
#include <gtest/gtest.h>
#include "smpl/smpl.h"

using mcs = std::chrono::microseconds;
using clk = std::chrono::system_clock;

smpl::SMPL model;

void func(int num_iter, int num_vert) {
    float theta[72] = {0}, beta[10] = {0};
    float *vert = (float *)malloc(3 * num_vert * sizeof(float));
    double *times = (double *)malloc(num_iter * sizeof(double));;

    for (int i = 0; i < 72; i++)
        theta[i] = rand();
    for (int i = 0; i < 10; i++)
        beta[i] = rand();

    for (int i = 0; i < 3 * num_vert; i++)
        vert[i] = rand();

    if (num_vert == 0)
        model.lbs_for_model(beta, theta);
    else
        model.lbs_for_custom_vertices(beta, theta, vert, num_vert);
    double sr = 0;
    for (int64_t i = 0; i < num_iter; i++) {
        auto begin = clk::now();
        if (num_vert == 0)
            model.lbs_for_model(beta, theta);
        else
            model.lbs_for_custom_vertices(beta, theta, vert, num_vert);
        auto end = clk::now();
        auto time = std::chrono::duration_cast<mcs>(end - begin);
        times[i] = (double)time.count();
        sr += times[i];
        std::cout << i << ") smpl " << times[i] << " sr: " << sr << std::endl;
    }
    sr /= num_iter;
    std::cout << "SR: " << sr << " mcs" << std::endl;

    double ds = 0;
    for (int64_t i = 0; i < num_iter; i++)
        ds += (times[i] - sr) * (times[i] - sr);
    ds /= num_iter;
    std::cout << "DISP: " << ds << " mcs" << std::endl;
}

int main(int argc, char const *argv[]) {
//    ::testing::InitGoogleTest(&argc, const_cast<char **>(argv));
//    return RUN_ALL_TESTS();
    auto begin = clk::now();
    std::string model_path = "/home/nponomareva/DoubleFusion/data/smpl_female.json";
    model.init(model_path);
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<mcs>(end - begin);
    std::cout << "Time duration to load SMPL: " << (double)duration.count() / 1000000 << " s" << std::endl;

    std::cout << "No reinit" << std::endl;
    func(1000, 0);

    std::cout << "No reinit, custom vertices" << std::endl;
    func(1000, 1000);

    return 0;
}