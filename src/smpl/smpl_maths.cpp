#include <fstream>
// #include <experimental/filesystem>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xjson.hpp>
#include "../../include/smpl/exception.h"
#include "../../include/smpl/smpl.h"
#include "../../include/def.h"

namespace smpl {

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    SMPL::blendShape(torch::Tensor &beta, torch::Tensor &theta, torch::Tensor &restTheta) {
        torch::Tensor poseRotation = rodrigues(theta);// (N, 24, 3, 3)
        torch::Tensor restPoseRotation;

        if (!restTheta.is_same(torch::Tensor())
            && restTheta.sizes() == torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3}))
            restPoseRotation = rodrigues(restTheta);// (N, 24, 3, 3)
        else {
            restPoseRotation = torch::eye(3, m__device);// (3, 3)
            restPoseRotation = restPoseRotation.expand({BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)
        }

        // unroll rotations
        torch::Tensor unPoseRot = torch::reshape(poseRotation, {BATCH_SIZE, JOINT_NUM * 3 * 3}),// (N, 216)
                unRestPoseRot = torch::reshape(restPoseRotation, {BATCH_SIZE, JOINT_NUM * 3 * 3});// (N, 216)
        // truncate rotations
        unPoseRot = TorchEx::indexing(unPoseRot, torch::IntList(), torch::IntList({9, unPoseRot.size(1)}));// (N, 207)
        unRestPoseRot = TorchEx::indexing(unRestPoseRot, torch::IntList(), torch::IntList({9, unRestPoseRot.size(1)}));// (N, 207)
        // pose blend coefficients
        torch::Tensor poseBlendCoeffs = unPoseRot - unRestPoseRot;// (N, 207)
        torch::Tensor poseBlendShape = torch::tensordot(poseBlendCoeffs, m__poseBlendBasis, {1}, {2});// (N, 6890, 3)
        torch::Tensor shapeBlendShape = torch::tensordot(beta, m__shapeBlendBasis, {1}, {2}); // (N, 6890, 3)

        return {poseRotation, restPoseRotation, poseBlendShape, shapeBlendShape};
    }

    std::tuple<torch::Tensor, torch::Tensor> SMPL::regressJoints(torch::Tensor &shapeBlendShape,
                                                                 torch::Tensor &poseBlendShape) {
        torch::Tensor restShape = m__templateRestShape + shapeBlendShape + poseBlendShape;
        torch::Tensor blendShape = m__templateRestShape + shapeBlendShape;// (N, 6890, 3)
        torch::Tensor joints = torch::tensordot(blendShape, m__jointRegressor, {1}, {1});// (N, 3, 24)
        joints = torch::transpose(joints, 1, 2);// (N, 24, 3)

        return {restShape, joints};
    }

    torch::Tensor SMPL::transform(torch::Tensor &poseRotation, torch::Tensor &joints) {
        torch::Tensor zeros = torch::zeros({BATCH_SIZE, JOINT_NUM, 1, 3}, m__device);// (N, 24, 1, 3)
        torch::Tensor poseRotHomo = torch::cat({poseRotation, zeros}, 2);// (N, 24, 4, 3)
        torch::Tensor localTransformations = localTransform(poseRotHomo, joints);
        torch::Tensor globalTransformations = globalTransform(localTransformations);

        torch::Tensor eliminated = torch::matmul(TorchEx::indexing(globalTransformations, torch::IntList(),
                                                                   torch::IntList(), torch::IntList({0, 3}),
                                                                   torch::IntList({0, 3})),
                                                 torch::unsqueeze(joints, 3));// (N, 24, 3, 1)
        zeros = torch::zeros({BATCH_SIZE, JOINT_NUM, 1, 1}, m__device);// (N, 24, 1, 1)
        torch::Tensor eliminatedHomo = torch::cat({eliminated, zeros}, 2);// (N, 24, 4, 1)
        zeros = torch::zeros({BATCH_SIZE, JOINT_NUM, 4, 3}, m__device);// (N, 24, 4, 3)
        eliminatedHomo = torch::cat({zeros, eliminatedHomo}, 3);// (N, 24, 4, 4)
        torch::Tensor transformation = globalTransformations - eliminatedHomo;// (N, 24, 4, 4)

        return transformation;
    }

    torch::Tensor SMPL::skinning(torch::Tensor &restShape, torch::Tensor &transformation) {
        // Cartesian coordinates to homogeneous coordinates
        torch::Tensor restShapeHomo = cart2homo(restShape);// (N, 6890, 4)
        // linear blend skinning
        torch::Tensor coefficients = torch::tensordot(m__weights, transformation, {1}, {1});// (6890, N, 4, 4)
        coefficients = torch::transpose(coefficients, 0, 1);// (N, 6890, 4, 4)
        restShapeHomo = torch::unsqueeze(restShapeHomo, 3);// (N, 6890, 4, 1)
        torch::Tensor verticesHomo = torch::matmul(coefficients, restShapeHomo);// (N, 6890, 4, 1)
        verticesHomo = torch::squeeze(verticesHomo, 3);// (N, 6890, 4)
        // homogeneous coordinates to Cartesian coordinates
        return homo2cart(verticesHomo);
    }

    torch::Tensor SMPL::rodrigues(torch::Tensor &theta) {
        // rotation angles and axis
        torch::Tensor angles = torch::norm(theta, 2, {2}, true);// (N, 24, 1)
        torch::Tensor axes = theta / angles;// (N, 24, 3)

        // skew symmetric matrices
        torch::Tensor zeros = torch::zeros({BATCH_SIZE, JOINT_NUM}, m__device);// (N, 24)
        torch::Tensor skew = torch::stack({zeros,
                                           -TorchEx::indexing(axes, torch::IntList(), torch::IntList(), torch::IntList({2})),
                                           TorchEx::indexing(axes, torch::IntList(), torch::IntList(), torch::IntList({1})),
                                           TorchEx::indexing(axes, torch::IntList(), torch::IntList(), torch::IntList({2})),
                                           zeros,
                                           -TorchEx::indexing(axes, torch::IntList(), torch::IntList(), torch::IntList({0})),
                                           -TorchEx::indexing(axes, torch::IntList(), torch::IntList(), torch::IntList({1})),
                                           TorchEx::indexing(axes, torch::IntList(), torch::IntList(), torch::IntList({0})),
                                           zeros}, 2);// (N, 24, 9)
        skew = torch::reshape(skew, {BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)

        // Rodrigues' formula
        torch::Tensor eye = torch::eye(3, m__device);// (3, 3)
        eye = eye.expand({BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)
        torch::Tensor sine = torch::sin(torch::unsqueeze(angles, 3).expand({BATCH_SIZE, JOINT_NUM, 3, 3}));// (N, 24, 3, 3)
        torch::Tensor cosine = torch::cos(torch::unsqueeze(angles, 3).expand({BATCH_SIZE, JOINT_NUM, 3, 3}));// (N, 24, 3, 3)
        torch::Tensor rotation = eye + skew * sine + torch::matmul(skew, skew) * (1 - cosine);// (N, 24, 3, 3)

        return rotation;
    }

    torch::Tensor SMPL::localTransform(torch::Tensor &poseRotHomo, torch::Tensor &joints) {
        std::vector<torch::Tensor> translations;
        translations.push_back(TorchEx::indexing(joints, torch::IntList(), torch::IntList({0}), torch::IntList()));// [0, (N, 3)]

        for (int64_t i = 1; i < JOINT_NUM; i++) {
            torch::Tensor ancestor = TorchEx::indexing(m__kinematicTree, torch::IntList({0}),
                                                       torch::IntList({i})).toType(torch::kLong);
            torch::Tensor translation = TorchEx::indexing(joints, torch::IntList(),
                                                          torch::IntList({i}), torch::IntList()) -
                                        torch::index_select(joints, 1, ancestor).squeeze(1);// (N, 3)
            translations.push_back(translation);// [i, (N, 3)]
        }
        torch::Tensor localTranslations = torch::unsqueeze(torch::stack(translations, 1), 3);// (N, 24, 3, 1)
        torch::Tensor ones = torch::ones({BATCH_SIZE, JOINT_NUM, 1, 1}, m__device);// (N, 24, 1, 1)
        torch::Tensor localTransformationsHomo = torch::cat({localTranslations, ones}, 2);// (N, 24, 4, 1)
        torch::Tensor localTransformations = torch::cat({poseRotHomo, localTransformationsHomo}, 3);// (N, 24, 4, 4)

        return localTransformations;
    }

    torch::Tensor SMPL::globalTransform(torch::Tensor &localTransformations) {
        std::vector<torch::Tensor> transformations;
        transformations.push_back(TorchEx::indexing(localTransformations, torch::IntList(), torch::IntList({0}),
                                                    torch::IntList(), torch::IntList()));// [0, (N, 4, 4)]

        for (int64_t i = 1; i < JOINT_NUM; i++) {
            torch::Tensor ancestor = TorchEx::indexing(m__kinematicTree, torch::IntArrayRef({0}),
                                                       torch::IntArrayRef({i})).toType(torch::kLong);
            torch::Tensor transformation = torch::matmul(
                    transformations[*(ancestor.to(torch::kCPU).data<int64_t>())],
                    TorchEx::indexing(localTransformations, torch::IntList(), torch::IntList({i}), torch::IntList(), torch::IntList())
            );
            transformations.push_back(transformation);// [i, (N, 4, 4)]
        }
        torch::Tensor globalTransformations = torch::stack(transformations, 1);// (N, 24, 4, 4)
        return globalTransformations;
    }

    torch::Tensor SMPL::cart2homo(torch::Tensor &cart) {
        torch::Tensor ones = torch::ones({BATCH_SIZE, VERTEX_NUM, 1}, m__device);// (N, 6890, 1)
        torch::Tensor homo = torch::cat({cart, ones}, 2);// (N, 6890, 4)
        return homo;
    }

    torch::Tensor SMPL::homo2cart(torch::Tensor &homo) {
        torch::Tensor homoW = TorchEx::indexing(homo, torch::IntList(), torch::IntList(), torch::IntList({3}));// (N, 6890)
        homoW = torch::unsqueeze(homoW, 2);// (N, 6890, 1)
        torch::Tensor homoUnit = homo / homoW;// (N, 6890, 4)
        torch::Tensor cart = TorchEx::indexing(homoUnit, torch::IntList(), torch::IntList(), torch::IntList({0, 3}));// (N, 6890, 3)
        return cart;
    }
}