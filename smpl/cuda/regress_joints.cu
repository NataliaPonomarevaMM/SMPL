#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void
        RegressJoints1(float *templateRestShape, float *shapeBlendShape, float *poseBlendShape, int vertexnum,
                       float *restShape) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            for (int k = 0; k < 3; k++) {
                int ind = i * vertexnum * 3 + j * 3 + k;
                restShape[ind] = templateRestShape[ind] + shapeBlendShape[ind] + poseBlendShape[ind];
            }
        }

        __global__ void
        RegressJoints2(float *templateRestShape, float *shapeBlendShape, float *jointRegressor, int jointnum,
                       int vertexnum,
                       float *joints) {
            int i = blockIdx.x;
            int j = threadIdx.x;
            for (int l = 0; l < 3; l++) {
                joints[i * jointnum * 3 + j * 3 + l] = 0;
                for (int k = 0; k < vertexnum; k++)
                    joints[i * jointnum * 3 + j * 3 + l] += (templateRestShape[i * vertexnum * 3 + k * 3 + l] +
                                                             shapeBlendShape[i * vertexnum * 3 + k * 3 + l]) *
                                                            jointRegressor[j * vertexnum + k];
            }
        }
    }

    std::tuple<float *, float *> SMPL::regressJoints(float *d_shapeBlendShape, float *d_poseBlendShape) {
        float *d_joints, *d_restShape;
        cudaMalloc((void **) &d_joints, BATCH_SIZE * JOINT_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_restShape, BATCH_SIZE * VERTEX_NUM * 3 * sizeof(float));

        device::RegressJoints1<<<BATCH_SIZE,VERTEX_NUM>>>(d_templateRestShape, d_shapeBlendShape, d_poseBlendShape,
                VERTEX_NUM, d_restShape);
        device::RegressJoints2<<<BATCH_SIZE,JOINT_NUM>>>(d_templateRestShape, d_shapeBlendShape, d_jointRegressor,
                JOINT_NUM, VERTEX_NUM, d_joints);

        return {d_restShape, d_joints};
    }
}