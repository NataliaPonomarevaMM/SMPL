#include <iostream>
#include <gtest/gtest.h>

int main(int argc, char const *argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char **>(argv));
    return RUN_ALL_TESTS();
    // return 0;
}