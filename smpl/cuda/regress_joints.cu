#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void
        RegressJoints1(float *templateRestShape, float *shapeBlendShape, float *poseBlendShape,
                       float *restShape) {
            int j = blockIdx.x;
            int k = threadIdx.x;

            int ind = j * 3 + k;
            restShape[ind] = templateRestShape[ind] + shapeBlendShape[ind] + poseBlendShape[ind];
        }

        __global__ void
        RegressJoints2(float *templateRestShape, float *shapeBlendShape, float *jointRegressor, int vertexnum,
                       float *joints) {
            int j = blockIdx.x;
            int l = threadIdx.x;

            int ind = j * 3 + l;
            joints[ind] = 0;
            for (int k = 0; k < vertexnum; k++)
                joints[ind] += (templateRestShape[k * 3 + l] +
                        shapeBlendShape[k * 3 + l]) * jointRegressor[j * vertexnum + k];
        }
    }

    std::tuple<float *, float *> SMPL::regressJoints(float *d_shapeBlendShape, float *d_poseBlendShape) {
        float *d_joints, *d_restShape;
        cudaMalloc((void **) &d_joints, JOINT_NUM * 3 * sizeof(float));
        cudaMalloc((void **) &d_restShape, VERTEX_NUM * 3 * sizeof(float));

        device::RegressJoints1<<<VERTEX_NUM,3>>>(d_templateRestShape, d_shapeBlendShape, d_poseBlendShape, d_restShape);
        device::RegressJoints2<<<JOINT_NUM,3>>>(d_templateRestShape, d_shapeBlendShape, d_jointRegressor,
                VERTEX_NUM, d_joints);

        return {d_restShape, d_joints};
    }
}