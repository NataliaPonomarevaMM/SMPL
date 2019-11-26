#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void PoseBlend1(float *theta, int jointnum,
                                   float *poseRotation, float *restPoseRotation) {
            int i = blockIdx.x;
            int j = threadIdx.x;

            int ind = i * jointnum * 3 + j * 3;
            float norm = std::sqrt(
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

            ind = ind * 3;
            for (int p = 0; p < 0; p++)
                poseRotation[ind + p] = 0;
            poseRotation[ind] = 1;
            poseRotation[ind + 4] = 1;
            poseRotation[ind + 8] = 1;
            for (int k1 = 0; k1 < 3; k1++)
                for (int k2 = 0; k2 < 3; k2++) {
                    int k = k1 * 3 + k2;
                    poseRotation[ind + k] += skew[k] * sin;
                    float num = 0;
                    for (int l = 0; l < 3; l++)
                        num += skew[k1 * 3 + l] * skew[l * 3 + k2];
                    poseRotation[ind + k] += (1 - cos) * num;// (N, 24, 3, 3)
                }

            for (int k = 0; k < 9; k++)
                restPoseRotation[ind + k] = 0;
            restPoseRotation[ind] = 1;
            restPoseRotation[ind + 4] = 1;
            restPoseRotation[ind + 8] = 1;
        }

        __global__ void
        PoseBlend2(float *poseRotation, float *poseBlendBasis, float *restPoseRotation, int jointnum, int vertexnum,
                   float *poseBlendShape) {
            int i = blockIdx.x; // batch size
            int j = threadIdx.x; // vertex num
            for (int k = 0; k < 3; k++) {
                poseBlendShape[i * vertexnum * 3 + j * 3 + k] = 0;
                for (int l = 0; l < 207; l++)
                    poseBlendShape[i * vertexnum * 3 + j * 3 + k] +=
                            (poseRotation[i * jointnum * 9 + l + 9] - restPoseRotation[i * jointnum * 9 + l + 9])
                            * poseBlendBasis[j * 3 * 207 + k * 207 + l];
            }
        }

        __global__ void ShapeBlend(float *beta, float *shapeBlendBasis, int vertexnum, int shapebasisdim,
                                   float *shapeBlendShape) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            for (int k = 0; k < 3; k++) {
                shapeBlendShape[i * vertexnum * 3 + j * 3 + k] = 0;
                for (int l = 0; l < shapebasisdim; l++)
                    shapeBlendShape[i * vertexnum * 3 + j * 3 + k] += beta[i * shapebasisdim + l] *
                                                                      shapeBlendBasis[j * shapebasisdim * 3 +
                                                                                      k * shapebasisdim +
                                                                                      l];// (N, 6890, 3)
            }
        }
    }

    std::tuple<float *, float *, float *, float *> SMPL::blendShape(float *theta, float *beta) {
        float *d_theta, *d_poseRotation, *d_restPoseRotation, *d_poseBlendShape;
        cudaMalloc((void **) &d_theta, BATCH_SIZE * JOINT_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_poseRotation, BATCH_SIZE * JOINT_NUM * 9 * sizeof(float));
        cudaMalloc((void **) &d_restPoseRotation, BATCH_SIZE * JOINT_NUM * 9 * sizeof(float));
        cudaMalloc((void **) &d_poseBlendShape, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));
        cudaMemcpy(d_theta, theta, BATCH_SIZE * JOINT_NUM * 3 * sizeof(float), cudaMemcpyHostToDevice);

        device::PoseBlend1<<<BATCH_SIZE,JOINT_NUM>>>(d_theta, JOINT_NUM, d_poseRotation, d_restPoseRotation);
        device::PoseBlend2<<<BATCH_SIZE,VERTEX_NUM>>>(d_poseRotation, d_poseBlendBasis, d_restPoseRotation,
                JOINT_NUM, VERTEX_NUM, d_poseBlendShape);

        float *d_beta, *d_shapeBlendShape;
        cudaMalloc((void **) &d_beta, BATCH_SIZE * SHAPE_BASIS_DIM * sizeof(float));
        cudaMalloc((void **) &d_shapeBlendShape, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));
        cudaMemcpy(d_beta, beta, BATCH_SIZE * SHAPE_BASIS_DIM * sizeof(float), cudaMemcpyHostToDevice);

        device::ShapeBlend<<<BATCH_SIZE,VERTEX_NUM>>>(d_beta, d_shapeBlendBasis, VERTEX_NUM, SHAPE_BASIS_DIM, d_shapeBlendShape);

        cudaFree(d_theta);
        cudaFree(d_beta);

        return {d_poseRotation, d_restPoseRotation, d_poseBlendShape, d_shapeBlendShape};
    }
}