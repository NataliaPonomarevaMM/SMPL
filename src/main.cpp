#include <torch/torch.h>
#include <iostream>
#include <gtest/gtest.h>
#include "../include/def.h"
#include "../include/smpl/smpl.h"

int main(int argc, char const *argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char **>(argv));
    return RUN_ALL_TESTS();
    // return 0;
}