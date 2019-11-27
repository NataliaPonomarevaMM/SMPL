#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void FindKNN1(float *templateRestShape, float *shapeBlendShape, int vertexnum, float *curvertices,
                                 float *dist) {
            //int i = blockIdx.x;
            int i = 0;
            int j = threadIdx.x;
            int ind = i * vertexnum + j;
            dist[ind] = 0;
            for (int k = 0; k < 3; k++) {
                float restShape = templateRestShape[j * 3 + k] + shapeBlendShape[j * 3 + k];
                dist[ind] += (curvertices[i * 3 + k] - restShape) * (curvertices - restShape);
            }
        }

        __global__ void FindKNN2(float *dist, int vertexnum,
                                 int *ind) {
            //int i = threadIdx.x;

            //sort them
            ind[0] = 0;
            ind[1] = 1;
            ind[2] = 2;
            ind[3] = 3;

            for (l = 0; l < 4; l++)
                for (p = 0; p < 3 - l; p++)
                    if (dist[ind[j]] > array[ind[j + 1]) {
                        int tmp = ind[j];
                        ind[j] = ind[j + 1];
                        ind[j + 1] = tmp;
                    }

            //find first 4 minimum distances
            for (int k = 4; k < vertexnum; k++)
                for (int t = 0; t < 4; t++)
                    if (dist[k] < dist[ind[t]]) {
                        for (int l = 3; l > t; l--)
                            ind[l] = ind[l - 1];
                        ind[t] = k;
                        continue;
                    }
        }

        __global__ void CalculateWeights(float *dist, float *weights, float *ind, int jointnum, int vertexnum,
                                         float *new_weights) {
            int j = threadIdx.x; // num of weight
            //int i = blockIdx.x;
            int i = 0; // num of vertex

            new_weights[i * jointnum + j] = 0;
            float weight = 0;
            for (int k = 0; k < 4; k++) {
                weight += dist[ind[i * 4 + k]];
                new_weights[i * jointnum + j] += dist[ind[i * 4 + k]] *
                        weights[ind[i * 4 + k] * jointnum + j]
            }
            new_weights[i * jointnum + j] /= weight;
        }
    }

    // linear blend skinning for vertex [3]
    float *SMPL::LBS(float *beta, float *theta, float *vertex) {
        auto bs = blendShape(beta, theta);
        auto d_poseRotation = std::get<0>(bs);
        auto d_restPoseRotation = std::get<1>(bs);
        auto d_poseBlendShape = std::get<2>(bs);
        auto d_shapeBlendShape = std::get<3>(bs);

        auto rj = regressJoints(d_shapeBlendShape, d_poseBlendShape);
        auto d_restShape = std::get<0>(rj);
        auto d_joints = std::get<1>(rj);
        cudaFree(d_poseBlendShape);

        auto d_transformation = transform(d_poseRotation, d_joints);
        cudaFree(d_poseRotation);
        cudaFree(d_joints);

        float *d_dist;
        cudaMalloc((void **) &d_dist, VERTEX_NUM * sizeof(float));
        float *d_ind;
        cudaMalloc((void **) &d_ind, 4 * sizeof(float));
        float *d_vertex;
        cudaMalloc((void **) &d_vertex, 3 * sizeof(float));
        cudaMemcpy(d_vertex, vertex, 3 * sizeof(float), cudaMemcpyHostToDevice);
        float *d_cur_weights;
        cudaMalloc((void **) &d_cur_weights, JOINT_NUM * sizeof(float));

        // find k nearest neigbours
        device::FindKNN1<<<1,VERTEX_NUM>>>(d_templateRestShape, d_shapeBlendShape, VERTEX_NUM, d_vertex, d_dist);
        device::FindKNN2<<<1,1>>>(d_dist, VERTEX_NUM, d_ind);
        //now we can calculate weights
        device::CalculateWeights<<<1,JOINT_NUM>>>(d_dist, m__weights, d_ind,  JOINT_NUM, VERTEX_NUM, d_cur_weights);
        cudaFree(d_shapeBlendShape);
        cudaFree(d_dist);
        cudaFree(d_ind);

        ///SKINNING
        float *d_vertices;
        cudaMalloc((void **) &d_vertices, 3 * sizeof(float));

        device::Skinning<<<BATCH_SIZE,VERTEX_NUM>>>(d_vertex, d_transformation, d_cur_weights, 1, 1, JOINT_NUM, d_vertices);

        float *result_vertices = (float *)malloc(3 * sizeof(float));
        cudaMemcpy(result_vertices, d_vertices, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vertices);
        return result_vertices;
    }
}