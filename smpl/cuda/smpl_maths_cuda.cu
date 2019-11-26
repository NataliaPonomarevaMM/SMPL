#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    void SMPL::loadToDevice() {
        ///BLEND SHAPE
        cudaMalloc((void **) &d_poseBlendBasis, VERTEX_NUM * 3 * POSE_BASIS_DIM * sizeof(float));
        cudaMemcpy(d_poseBlendBasis, m__poseBlendBasis, VERTEX_NUM * 3 * POSE_BASIS_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &d_shapeBlendBasis, VERTEX_NUM * 3 * SHAPE_BASIS_DIM * sizeof(float));
        cudaMemcpy(d_shapeBlendBasis, m__shapeBlendBasis, VERTEX_NUM * 3 * SHAPE_BASIS_DIM * sizeof(float), cudaMemcpyHostToDevice);
        ///REGRESS JOINTS
        cudaMalloc((void **) &d_templateRestShape, VERTEX_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_jointRegressor, JOINT_NUM * VERTEX_NUM * sizeof(float));
        cudaMemcpy(d_templateRestShape, m__templateRestShape, VERTEX_NUM * 3  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jointRegressor, m__jointRegressor, JOINT_NUM * VERTEX_NUM * sizeof(float), cudaMemcpyHostToDevice);
        ///WORLD TRANSFORMATIONS
        cudaMalloc((void **) &d_kinematicTree, 2 * JOINT_NUM * sizeof(int64_t));
        cudaMemcpy(d_kinematicTree, m__kinematicTree, 2 * JOINT_NUM * sizeof(int), cudaMemcpyHostToDevice);
        ///SKINNING
        cudaMalloc((void **) &d_weights, VERTEX_NUM * JOINT_NUM * sizeof(float));
        cudaMemcpy(d_weights, m__weights, VERTEX_NUM * JOINT_NUM * sizeof(float), cudaMemcpyHostToDevice);
    }

    float *SMPL::run(float *beta, float *theta) {
        auto bs = blendShape(beta, theta);
        auto d_poseRotation = std::get<0>(bs);
        auto d_restPoseRotation = std::get<1>(bs);
        auto d_poseBlendShape = std::get<2>(bs);
        auto d_shapeBlendShape = std::get<3>(bs);

        auto rj = regressJoints(d_shapeBlendShape, d_poseBlendShape);
        auto d_restShape = std::get<0>(rj);
        auto d_joints = std::get<1>(rj);
        //auto [d_poseRotation, d_restPoseRotation, d_poseBlendShape, d_shapeBlendShape] = blendShape(beta, theta);
        //auto [d_restShape, d_joints] = regressJoints(d_shapeBlendShape, d_poseBlendShape);
        cudaFree(d_shapeBlendShape);
        cudaFree(d_poseBlendShape);
        auto d_transformation = transform(d_poseRotation, d_joints);
        cudaFree(d_poseRotation);
        cudaFree(d_joints);
        float *res = skinning(d_restShape, d_transformation);
        cudaFree(d_restShape);
        cudaFree(d_transformation);

        return res;
    }

    SMPL::~SMPL() {
        ///CPU
        if (m__faceIndices != nullptr)
            free(m__faceIndices);
        if (m__shapeBlendBasis != nullptr)
            free(m__shapeBlendBasis);
        if (m__poseBlendBasis != nullptr)
            free(m__poseBlendBasis);
        if (m__templateRestShape != nullptr)
            free(m__templateRestShape);
        if (m__jointRegressor != nullptr)
            free(m__jointRegressor);
        if (m__kinematicTree != nullptr)
            free(m__kinematicTree);
        if (m__weights != nullptr)
            free(m__weights);

        ///GPU
        if (d_poseBlendBasis != nullptr)
            cudaFree(d_poseBlendBasis);
        if (d_shapeBlendBasis != nullptr)
            cudaFree(d_shapeBlendBasis);
        if (d_templateRestShape != nullptr)
            cudaFree(d_templateRestShape);
        if (d_jointRegressor != nullptr)
            cudaFree(d_jointRegressor);
        if (d_weights != nullptr)
            cudaFree(d_weights);
        if (d_kinematicTree != nullptr)
            cudaFree(d_kinematicTree);
    }
}