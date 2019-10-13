#include <fstream>
#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>
#include <iostream>
#include <xtensor/xjson.hpp>
#include <gtest/gtest.h>
#include "../../include/def.h"
#include "../../include/smpl/smpl.h"

namespace smpl {
//    TEST_F(SMPL, Import) {
//        std::ifstream file("~/DoubleFusion/data/smpl_female.json");
//
//        nlohmann::json model;
//        file >> model;
//
//        xt::xarray<int64_t> kinematicTree;
//        xt::from_json(model["kinematic_tree"], kinematicTree);
//
//        int64_t r_kinematicTree[2][24] = {
//                {4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  9,  9,  12, 13, 14, 16, 17, 18, 19, 20, 21},
//                {0,          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}};
//        for (int i = 0; i < 2; i++)
//            for (int j = 0; j < JOINT_NUM; j++)
//                ASSERT_EQ(r_kinematicTree[i][j], kinematicTree.at(i, j));
//    }
}
