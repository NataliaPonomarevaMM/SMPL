#include <cmath>
#include "def.h"
#include "../../include/smpl/smpl.h"

namespace smpl {
    namespace device {
        __global__ void PoseBlend1(float *theta,
                                   float *poseRotation, float *restPoseRotation) {
            int i = blockIdx.x;
            int j = threadIdx.x;

            int ind = i * JOINT_NUM * 3 + j * 3;
            float norm = (theta[ind] + theta[ind + 1] + theta[ind + 2]) / std::sqrt(
                    theta[ind] * theta[ind] + theta[ind + 1] * theta[ind + 1] + theta[ind + 2] * theta[ind + 2]);
            float sin = std::sin(norm);
            float cos = std::cos(norm);
            theta[ind] /= norm;
            theta[ind + 1] /= norm;
            theta[ind + 2] /= norm; // axes

            float skew[9];
            skew[0] = 0;
            skew[1] = -1 * theta[ind + 2];
            skew[2] = theta[ind + 1];
            skew[3] = theta[ind + 2];
            skew[4] = 0;
            skew[5] = -1 * theta[ind];
            skew[6] = -1 * theta[ind + 1];
            skew[7] = theta[ind];
            skew[8] = 0;

            poseRotation[ind] = 1;
            poseRotation[ind + 4] = 1;
            poseRotation[ind + 8] = 1;
            for (int k1 = 0; k1 < 3; k1++)
                for (int k2 = 0; k2 < 3; k2++) {
                    int k = k1 * 3 + k2;
                    poseRotation[ind + k] += skew[k] * sin;
                    float num = 0;
                    for (int l = 0; l < 3; l++)
                        num += skew[k1 * 3 + l] * skew[k * 3 + k2];
                    poseRotation[ind + k] += (1 - cos) * num;// (N, 24, 3, 3)
                }

            for (int k = 0; k < 9; k++)
                restPoseRotation[i * JOINT_NUM * 9 + j * 9 + k] = 0;
            restPoseRotation[i * JOINT_NUM * 9 + j * 9] = 1;
            restPoseRotation[i * JOINT_NUM * 9 + j * 9 + 4] = 1;
            restPoseRotation[i * JOINT_NUM * 9 + j * 9 + 8] = 1;
        }

        __global__ void PoseBlend2(float *poseRotation, float *poseBlendBasis, float *restPoseRotation,
                                   float *poseBlendShape) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            for (int k = 0; k < 3; k++) {
                poseBlendShape[i * VERTEX_NUM * 3 + j * 3 + k] = 0;
                for (int l = 0; l < 207; l++)
                    poseBlendShape[i * VERTEX_NUM * 3 + j * 3 + k] +=
                            (poseRotation[i * JOINT_NUM * 9 + l + 9] - restPoseRotation[i * JOINT_NUM * 9 + l + 9])
                                                * poseBlendBasis[j * VERTEX_NUM * 3 + k * 3 + l];
            }
        }

        __global__ void ShapeBlend(float *beta, float *shapeBlendBasis,
                                   float *shapeBlendShape) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            for (int k = 0; k < 3; k++) {
                shapeBlendShape[i * VERTEX_NUM * 3 + j * 3 + k] = 0;
                for (int l = 0; l < SHAPE_BASIS_DIM; l++)
                    shapeBlendShape[i * VERTEX_NUM * 3 + j * 3 + k] += beta[i * SHAPE_BASIS_DIM + l] *
                            shapeBlendBasis[j * SHAPE_BASIS_DIM * 3 + k * SHAPE_BASIS_DIM + l];// (N, 6890, 3)
            }
        }

        __global__ void RegressJoints(float *templateRestShape, float *shapeBlendShape, float *poseBlendShape, float *jointRegressor,
                                      float *joints, float *restShape) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            for (int k = 0; k < 3; k++) {
                int ind = i * VERTEX_NUM * 3 + j * 3 + k;
                restShape[ind] = templateRestShape[ind] + shapeBlendShape[ind] + poseBlendShape[ind];
            }
            for (int k = 0; k < JOINT_NUM; k++)
                for (int l = 0; l < 3; l++)
                    joints[i * JOINT_NUM * 3+ k * 3 + l] += (templateRestShape[i * VERTEX_NUM * 3 + j * 3 + l] +
                            shapeBlendShape[i * VERTEX_NUM * 3 + j * 3 + l]) * jointRegressor[k * VERTEX_NUM + j];
        }

        __global__ void LocalTransform(float *joints, float *kinematicTree, float *poseRotation,
                                       float *localTransformations) {
            // joints [BATCH_SIZE][JOINT_NUM][3]
            // poseRotHomo [BATCH_SIZE][JOINTS_NUM][4][3]
            // kinematicTree [2][JOINT_NUM]
            int j = blockIdx.x;
            int i = threadIdx.x;
            //copy data from poseRotation
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    localTransformations[j * JOINT_NUM * 16 + i * 16 + k * 4 + l] =
                            poseRotation[j * JOINT_NUM * 9 + i * 9 + k * 3 + l];
            for (int l = 0; l < 3; l++)
                localTransformations[j * JOINT_NUM * 16 + i * 16 + 3 * 4 + l] = 0;
            // data from joints
            int ancestor = kinematicTree[i];
            for (int k = 0; k < 3; k++)
                localTransformations[j * JOINT_NUM * 16 + i * 16 + k * 4 + 3] =
                        i == 0 ? joints[j * JOINT_NUM * 3 + i * 3 + k] - joints[j * JOINT_NUM * 3 + ancestor * 3 + k]
                        : joints[j * JOINT_NUM * 3 + k];
            localTransformations[j * JOINT_NUM * 16 + i * 16 + 3 * 4 + 3] = 1;
        }


        __global__ void GlobalTransform(float *localTransformations, float *kinematicTree,
                                        float *globalTransformations) {
            //global transformations [N][24][4][4]
            for (int i = 0; i < BATCH_SIZE; i++)
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++)
                        globalTransformations[i * JOINT_NUM * 16 + k * 4 + l] = localTransformations[i * JOINT_NUM * 16 + k * 4 + l];
            for (int j = 1; j < JOINT_NUM; j++) {
                int anc = kinematicTree[j];
                for (int i = 0; i < BATCH_SIZE; i++)
                    for (int k = 0; k < 4; k++)
                        for (int l = 0; l < 4; l++) {
                            globalTransformations[i * JOINT_NUM * 16 + j * 16 + k * 4 + l] = 0;
                            for (int t = 0; t < 4; t++)
                                globalTransformations[i * JOINT_NUM * 16 + j * 16 + k * 4 + l] +=
                                        globalTransformations[i * JOINT_NUM * 16 + anc * 16 + k * 4 + t] *
                                        localTransformations[i * JOINT_NUM * 16 + j * 16 + t * 4 + l];
                        }
            }
        }

        __global__ void Transform(float *globalTransformations, float *joints) {
            int i = blockIdx.x;
            int j = threadIdx.x;

            float elim[3] = {0};
            for (int k = 0; k < 3; k++)
                for (int t = 0; t < 3; t++)
                    elim[k] += globalTransformations[i * JOINT_NUM * 16 + j * 16 + k * 4 + t * 4] *
                            joints[i * JOINT_NUM * 3 + j * 3 + t];
            for (int k = 0; k < 3; k++)
                globalTransformations[i * JOINT_NUM * 16 + j * 16 + k * 4] -= elim[k];
        }

        torch::Tensor SMPL::cart2homo(torch::Tensor &cart) {
            torch::Tensor ones = torch::ones({BATCH_SIZE, VERTEX_NUM, 1}, m__device);// (N, 6890, 1)
            torch::Tensor homo = torch::cat({cart, ones}, 2);// (N, 6890, 4)
            return homo;
        }

        __global__ void Skinning(float *restShape, float *transformation, float *weights
                                 float *vertices) {
            //restShape [BATCH_SIZE][VERTEX_NUM][3]
            //transformation [BATCH_SIZE][JOINT_NUM][4][4]
            //weights [VERTEX_NUM][JOINT_NUM]

            // linear blend skinning
            for (int i = 0; i < BATCH_SIZE; i++)
                for (int j = 0; j < VERTEX_NUM; j++) {
                    float coeffs[16] = {0};
                    for (int k = 0; k < 4; k++)
                        for (int l = 0; l < 4; l++)
                            for (int t = 0; t < JOINT_NUM; t++)
                                coeffs[k * 4 + l] +=weights[j * JOINT_NUM + t] * transformation[i * JOINT_NUM * 16 + t * 16 + k * 4 + l];

                    float homoW = coeffs[15];
                    for (int t = 0; t < 3; t++)
                        homoW += coeffs[12 + t] * restShape[i * VERTEX_NUM * 3 + j * 3 + t];
                    for (int k = 0; k < 3; k++) {
                        vertices[i * VERTEX_NUM * 3 + j * 3 + k] = coeffs[k * 4 + 3];
                        for (int t = 0; t < 3; t++)
                            vertices[i * VERTEX_NUM * 3 + j * 3 + k] += coeffs[k * 4 + t] * restShape[i * VERTEX_NUM * 3 + j * 3 + t];
                        vertices[i * VERTEX_NUM * 3 + j * 3 + k] /= homoW;
                    }
                }
        }
    }

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

    void SMPL::run(float *beta, float *theta) {
        auto [d_poseRotation, d_restPoseRotation, d_poseBlendShape, d_shapeBlendShape] = blendShape(beta, theta);
        auto [d_restShape, d_joints] = regressJoints(d_shapeBlendShape, d_poseBlendShape);
        cudaFree(d_shapeBlendShape);
        cudaFree(d_poseBlendShape);
        auto d_transformation = transform(d_poseRotation, d_joints);
        cudaFree(d_poseRotation);
        cudaFree(d_joints);
        skinning(d_restShape, d_transformation);
        cudaFree(d_restShape);
        cudaFree(d_transformation);
    }

    std::tuple<float *, float *, float *, float *> SMPL::blendShape(float *theta, float *beta) {
        ///BLEND SHAPE
        float *d_theta, *d_poseRotation, *d_restPoseRotation, *d_poseBlendShape;
        cudaMalloc((void **) &d_theta, BATCH_SIZE * JOINT_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_poseRotation, BATCH_SIZE * JOINT_NUM * 9 * sizeof(float));
        cudaMalloc((void **) &d_restPoseRotation, BATCH_SIZE * JOINT_NUM * 9 * sizeof(float));
        cudaMalloc((void **) &d_poseBlendShape, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));
        cudaMemcpy(d_theta, theta, BATCH_SIZE * JOINT_NUM * 3 * sizeof(float), cudaMemcpyHostToDevice);

        device::PoseBlend1<<<BATCH_SIZE,JOINT_NUM>>>(d_theta, d_poseRotation, d_restPoseRotation);
        device::PoseBlend2<<<BATCH_SIZE,VERTEX_NUM>>>(d_poseRotation, d_poseBlendBasis, d_restPoseRotation, d_poseBlendShape);

        float *d_beta, *d_shapeBlendShape;
        cudaMalloc((void **) &d_beta, BATCH_SIZE * SHAPE_BASIS_DIM * sizeof(float));
        cudaMalloc((void **) &d_shapeBlendShape, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));
        cudaMemcpy(d_beta, beta, BATCH_SIZE * SHAPE_BASIS_DIM * sizeof(float), cudaMemcpyHostToDevice);

        device::ShapeBlend<<<BATCH_SIZE,VERTEX_NUM>>>(d_beta, d_shapeBlendBasis, d_shapeBlendShape);

        cudaFree(d_theta);
        cudaFree(d_beta);

        return {d_poseRotation, d_restPoseRotation, d_poseBlendShape, d_shapeBlendShape};
    }

    std::tuple<float *, float *> SMPL::regressJoints(float *d_shapeBlendShape, float *d_poseBlendShape) {
        ///REGRESS JOINTS
        float *d_joints, *d_restShape;
        cudaMalloc((void **) &d_joints, BATCH_SIZE * JOINT_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_restShape, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));

        device::RegressJoints<<<BATCH_SIZE,VERTEX_NUM>>>(d_templateRestShape, d_shapeBlendShape, d_poseBlendShape,
                d_jointRegressor, d_joints, d_restShape);

        return {d_restShape, d_joints};
    }

    float *SMPL::transform(float *d_poseRotation, float *d_joints) {
        ///WORLD TRANSFORMATIONS
        float *d_localTransformations, *d_globalTransformations;
        cudaMalloc((void **) &d_localTransformations, BATCH_SIZE * JOINT_NUM * 16 * sizeof(float));
        cudaMalloc((void **) &d_globalTransformations, BATCH_SIZE * JOINT_NUM * 16 * sizeof(float));

        device::LocalTransform<<<BATCH_SIZE,JOINT_NUM>>>(d_joints, d_kinematicTree, d_poseRotation, d_localTransformations);
        device::GlobalTransform<<<1,1>>>(d_localTransformations, d_kinematicTree, d_globalTransformations);
        device::GlobalTransform<<<BATCH_SIZE,JOINT_NUM>>>(d_globalTransformations, d_joints);

        cudaFree(d_localTransformations);
        return d_globalTransformations;
    }

    void SMPL::skinning(float *restShape, float *transformation) {
        ///SKINNING
        float *d_vertices;
        cudaMalloc((void **) &d_vertices, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));

        device::Skinning<<<BATCH_SIZE,VERTEX_NUM>>>(d_restShape, d_globalTransformations, d_weights, d_vertices);

        cudaMemcpy(m__result_vertices, d_vertices, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vertices);
    }

    SMPL::~SMPL() {
        //CPU
        free(m__faceIndices);
        free(m__shapeBlendBasis);
        free(m__poseBlendBasis);
        free(m__templateRestShape);
        free(m__jointRegressor);
        free(m__kinematicTree);
        free(m__weights);
        free(m__result_vertices);

        ///GPU
        cudaFree(d_poseBlendBasis);
        cudaFree(d_shapeBlendBasis);
        cudaFree(d_templateRestShape);
        cudaFree(d_jointRegressor);
        cudaFree(d_weights);
        cudaFree(d_kinematicTree);
    }
}