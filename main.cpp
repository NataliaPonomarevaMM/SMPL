#include <chrono>
#include <torch/torch.h>
#include <iostream>
#include "def.h"
#include "smpl/SMPL.h"
#include "tests/Singleton.h"

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

int main(int argc, char const *argv[]) {
    torch::Device cuda(torch::kCUDA);
    cuda.set_index(0);

    std::string modelPath = "../data/smpl_female.json";
    std::string outputPath = "../out/vertices.obj";

    torch::Tensor beta = 0.03 * torch::rand(
            {BATCH_SIZE, SHAPE_BASIS_DIM});// (N, 10)
    torch::Tensor theta = 0.2 * torch::rand(
            {BATCH_SIZE, JOINT_NUM, 3});// (N, 24, 3)

    torch::Tensor vertices;
    smpl::Singleton<smpl::SMPL>::get()->setDevice(cuda);
    smpl::Singleton<smpl::SMPL>::get()->setModelPath(modelPath);

    auto begin = clk::now();
    smpl::Singleton<smpl::SMPL>::get()->init();
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end - begin);
    std::cout << "Time duration to load SMPL: "
              << (double)duration.count() / 1000 << " s" << std::endl;

    const int64_t LOOPS = 100;
    duration = std::chrono::duration_cast<ms>(end - end);// reset duration
    for (int64_t i = 0; i < LOOPS; i++) {
        begin = clk::now();
        smpl::Singleton<smpl::SMPL>::get()->launch(beta, theta);
        end = clk::now();
        duration += std::chrono::duration_cast<ms>(end - begin);
    }
    std::cout << "Time duration to run SMPL: "
              << (double)duration.count() / LOOPS << " ms" << std::endl;

    vertices = smpl::Singleton<smpl::SMPL>::get()->getVertex();

    smpl::Singleton<smpl::SMPL>::get()->setVertPath(outputPath);
    smpl::Singleton<smpl::SMPL>::get()->out(0);

    smpl::Singleton<smpl::SMPL>::destroy();
    return 0;
}