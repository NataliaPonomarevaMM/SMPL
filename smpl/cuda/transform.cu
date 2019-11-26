#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void LocalTransform(float *joints, int64_t *kinematicTree, float *poseRotation, int jointnum,
                                       float *localTransformations) {
            // joints [batchsize][jointnum][3]
            // poseRotHomo [batchsize][JOINTS_NUM][4][3]
            // kinematicTree [2][jointnum]
            int j = blockIdx.x;
            int i = threadIdx.x;
            //copy data from poseRotation
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    localTransformations[j * jointnum * 16 + i * 16 + k * 4 + l] =
                            poseRotation[j * jointnum * 9 + i * 9 + k * 3 + l];
            for (int l = 0; l < 3; l++)
                localTransformations[j * jointnum * 16 + i * 16 + 3 * 4 + l] = 0;
            // data from joints
            int ancestor = kinematicTree[i];
            for (int k = 0; k < 3; k++)
                localTransformations[j * jointnum * 16 + i * 16 + k * 4 + 3] =
                        i != 0 ? joints[j * jointnum * 3 + i * 3 + k] - joints[j * jointnum * 3 + ancestor * 3 + k]
                               : joints[j * jointnum * 3 + k];
            localTransformations[j * jointnum * 16 + i * 16 + 3 * 4 + 3] = 1;
        }


        __global__ void
        GlobalTransform(float *localTransformations, int64_t *kinematicTree, int jointnum, int batchsize,
                        float *globalTransformations) {
            //global transformations [N][24][4][4]
            for (int i = 0; i < batchsize; i++)
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++)
                        globalTransformations[i * jointnum * 16 + k * 4 + l] = localTransformations[i * jointnum * 16 +
                                                                                                    k * 4 + l];
            for (int j = 1; j < jointnum; j++) {
                int anc = kinematicTree[j];
                for (int i = 0; i < batchsize; i++)
                    for (int k = 0; k < 4; k++)
                        for (int l = 0; l < 4; l++) {
                            globalTransformations[i * jointnum * 16 + j * 16 + k * 4 + l] = 0;
                            for (int t = 0; t < 4; t++)
                                globalTransformations[i * jointnum * 16 + j * 16 + k * 4 + l] +=
                                        globalTransformations[i * jointnum * 16 + anc * 16 + k * 4 + t] *
                                        localTransformations[i * jointnum * 16 + j * 16 + t * 4 + l];
                        }
            }
        }

        __global__ void Transform(float *globalTransformations, float *joints, int jointnum) {
            int i = blockIdx.x;
            int j = threadIdx.x;

            float elim[3];
            for (int k = 0; k < 3; k++) {
                elim[k] = 0;
                for (int t = 0; t < 3; t++)
                    elim[k] += globalTransformations[i * jointnum * 16 + j * 16 + k * 4 + t] *
                               joints[i * jointnum * 3 + j * 3 + t];
            }
            for (int k = 0; k < 3; k++)
                globalTransformations[i * jointnum * 16 + j * 16 + k * 4 + 3] -= elim[k];
        }
    }

    float *SMPL::transform(float *d_poseRotation, float *d_joints) {
        float *d_localTransformations, *d_globalTransformations;
        cudaMalloc((void **) &d_localTransformations, BATCH_SIZE * JOINT_NUM * 16 * sizeof(float));
        cudaMalloc((void **) &d_globalTransformations, BATCH_SIZE * JOINT_NUM * 16 * sizeof(float));

        device::LocalTransform<<<BATCH_SIZE,JOINT_NUM>>>(d_joints, d_kinematicTree, d_poseRotation, JOINT_NUM, d_localTransformations);
        device::GlobalTransform<<<1,1>>>(d_localTransformations, d_kinematicTree, JOINT_NUM, BATCH_SIZE, d_globalTransformations);
        device::Transform<<<BATCH_SIZE,JOINT_NUM>>>(d_globalTransformations, d_joints, JOINT_NUM);

        cudaFree(d_localTransformations);
        return d_globalTransformations;
    }
}