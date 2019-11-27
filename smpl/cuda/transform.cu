#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void LocalTransform(float *joints, int64_t *kinematicTree, float *poseRotation,
                                       float *localTransformations) {
            // joints [jointnum][3]
            // poseRotHomo [JOINTS_NUM][4][3]
            // kinematicTree [2][jointnum]
            int i = threadIdx.x;
            //copy data from poseRotation
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    localTransformations[i * 16 + k * 4 + l] = poseRotation[i * 9 + k * 3 + l];
            for (int l = 0; l < 3; l++)
                localTransformations[i * 16 + 3 * 4 + l] = 0;
            // data from joints
            int ancestor = kinematicTree[i];
            for (int k = 0; k < 3; k++)
                localTransformations[i * 16 + k * 4 + 3] = i != 0 ? joints[i * 3 + k] - joints[ancestor * 3 + k] : joints[k];
            localTransformations[i * 16 + 3 * 4 + 3] = 1;
        }


        __global__ void
        GlobalTransform(float *localTransformations, int64_t *kinematicTree, int jointnum,
                        float *globalTransformations) {
            //global transformations [N][24][4][4]
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    globalTransformations[k * 4 + l] = localTransformations[k * 4 + l];

            for (int j = 1; j < jointnum; j++) {
                int anc = kinematicTree[j];
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++) {
                        globalTransformations[j * 16 + k * 4 + l] = 0;
                        for (int t = 0; t < 4; t++)
                            globalTransformations[j * 16 + k * 4 + l] +=
                                    globalTransformations[anc * 16 + k * 4 + t] *
                                    localTransformations[j * 16 + t * 4 + l];
                    }
            }
        }

        __global__ void Transform(float *globalTransformations, float *joints) {
            int j = threadIdx.x;

            float elim[3];
            for (int k = 0; k < 3; k++) {
                elim[k] = 0;
                for (int t = 0; t < 3; t++)
                    elim[k] += globalTransformations[j * 16 + k * 4 + t] * joints[j * 3 + t];
            }
            for (int k = 0; k < 3; k++)
                globalTransformations[j * 16 + k * 4 + 3] -= elim[k];
        }
    }

    float *SMPL::transform(float *d_poseRotation, float *d_joints) {
        float *d_localTransformations, *d_globalTransformations;
        cudaMalloc((void **) &d_localTransformations, JOINT_NUM * 16 * sizeof(float));
        cudaMalloc((void **) &d_globalTransformations, JOINT_NUM * 16 * sizeof(float));

        device::LocalTransform<<<1,JOINT_NUM>>>(d_joints, d_kinematicTree, d_poseRotation, d_localTransformations);
        device::GlobalTransform<<<1,1>>>(d_localTransformations, d_kinematicTree, JOINT_NUM, d_globalTransformations);
        device::Transform<<<1,JOINT_NUM>>>(d_globalTransformations, d_joints, JOINT_NUM);

        cudaFree(d_localTransformations);
        return d_globalTransformations;
    }
}