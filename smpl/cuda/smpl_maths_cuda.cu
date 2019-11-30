#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    void SMPL::loadToDevice(float *shapeBlendBasis, float *poseBlendBasis,
            float *templateRestShape, float *jointRegressor,
            int64_t *kinematicTree, float *weights) {
        ///BLEND SHAPE
        cudaMalloc((void **) &d_poseBlendBasis, VERTEX_NUM * 3 * POSE_BASIS_DIM * sizeof(float));
        cudaMemcpy(d_poseBlendBasis, poseBlendBasis, VERTEX_NUM * 3 * POSE_BASIS_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &d_shapeBlendBasis, VERTEX_NUM * 3 * SHAPE_BASIS_DIM * sizeof(float));
        cudaMemcpy(d_shapeBlendBasis, shapeBlendBasis, VERTEX_NUM * 3 * SHAPE_BASIS_DIM * sizeof(float), cudaMemcpyHostToDevice);
        ///REGRESS JOINTS
        cudaMalloc((void **) &d_templateRestShape, VERTEX_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_jointRegressor, JOINT_NUM * VERTEX_NUM * sizeof(float));
        cudaMemcpy(d_templateRestShape, templateRestShape, VERTEX_NUM * 3  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jointRegressor, jointRegressor, JOINT_NUM * VERTEX_NUM * sizeof(float), cudaMemcpyHostToDevice);
        ///WORLD TRANSFORMATIONS
        cudaMalloc((void **) &d_kinematicTree, 2 * JOINT_NUM * sizeof(int64_t));
        cudaMemcpy(d_kinematicTree, kinematicTree, 2 * JOINT_NUM * sizeof(int), cudaMemcpyHostToDevice);
        ///SKINNING
        cudaMalloc((void **) &d_weights, VERTEX_NUM * JOINT_NUM * sizeof(float));
        cudaMemcpy(d_weights, weights, VERTEX_NUM * JOINT_NUM * sizeof(float), cudaMemcpyHostToDevice);
    }

    float *SMPL::run(float *beta, float *theta, float *d_custom_weights, float *d_vertices, float vertexnum) {
        auto pbs = poseBlendShape(theta);
        auto d_poseRotation = std::get<0>(pbs);
        auto d_restPoseRotation = std::get<1>(pbs);
        auto d_poseBlendShape = std::get<2>(pbs);

        auto d_shapeBlendShape = shapeBlendShape(beta);

        auto rj = regressJoints(d_shapeBlendShape, d_poseBlendShape);
        auto d_restShape = std::get<0>(rj);
        auto d_joints = std::get<1>(rj);
        cudaFree(d_shapeBlendShape);
        cudaFree(d_poseBlendShape);
        auto d_transformation = transform(d_poseRotation, d_joints);
        cudaFree(d_poseRotation);
        cudaFree(d_joints);

        if (d_vertices == nullptr) {
            d_vertices = d_restShape;
            vertexnum = VERTEX_NUM;
        }

        float *res = skinning(d_transformation, d_custom_weights, d_vertices, vertexnum);
        cudaFree(d_restShape);
        cudaFree(d_transformation);

        return res;
    }

    float *SMPL::lbs_for_model(float *beta, float *theta) {
        return run(beta, theta, d_weights);
    }

    SMPL::~SMPL() {
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