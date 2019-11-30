#include <iostream>
#include <cstdlib>
#include <chrono>
#include <gtest/gtest.h>
#include "smpl/smpl.h"

using mcs = std::chrono::microseconds;
using clk = std::chrono::system_clock;

smpl::SMPL model;

void func(bool with_reinit, int num_vert) {
    float theta[72] = {0}, beta[10] = {0};
    float *vert = (float *)malloc(3 * num_vert * sizeof(float));

    for (int i = 0; i < 72; i++)
        theta[i] = rand();
    for (int i = 0; i < 10; i++)
        beta[i] = rand();

    for (int i = 0; i < 3 * num_vert; i++)
        vert[i] = rand();

    double times[100];

    const int64_t LOOPS = 100;
    auto begin = clk::now();
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<mcs>(end - end);
    for (int64_t i = 0; i < LOOPS; i++) {
        if (with_reinit) {
            for (int i = 0; i < 72; i++)
                theta[i] = rand();
            for (int i = 0; i < 10; i++)
                beta[i] = rand();
        }

        begin = clk::now();
        if (num_vert == 0)
            model.lbs_for_model(beta, theta);
        else
            model.lbs_for_custom_vertices(beta, theta, vert, num_vert);
        end = clk::now();
        auto time = std::chrono::duration_cast<mcs>(end - begin);
        if (i != 0)
            duration += time;
        times[i] = (double)time.count();
        std::cout << i << ") smpl " << times[i] << std::endl;
    }
    double sr = (double)duration.count() / (LOOPS - 1.0);
    std::cout << "SR: " << sr << " mcs" << std::endl;

    double ds = 0;
    for (int64_t i = 1; i < LOOPS; i++)
        ds += (times[i] - sr) * (times[i] - sr);
    ds /= (LOOPS - 1.0);
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
    func(false, 0);
    std::cout << "With reinit" << std::endl;
    func(true, 0);

    std::cout << "No reinit, custom vertices" << std::endl;
    func(false, 10);

    return 0;
}