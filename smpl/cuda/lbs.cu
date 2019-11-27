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
                dist[ind] += (curvertices[i * 3 + k] - restShape) * (curvertices[i * 3 + k] - restShape);
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

            for (int l = 0; l < 4; l++)
                for (int p = 0; p < 3 - l; p++)
                    if (dist[ind[p]] > dist[ind[p + 1]]) {
                        int tmp = ind[p];
                        ind[p] = ind[p + 1];
                        ind[p + 1] = tmp;
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

        __global__ void CalculateWeights(float *dist, float *weights, int *ind, int jointnum, int vertexnum,
                                         float *new_weights) {
            int j = threadIdx.x; // num of weight
            //int i = blockIdx.x;
            int i = 0; // num of vertex

            new_weights[i * jointnum + j] = 0;
            float weight = 0;
            for (int k = 0; k < 4; k++) {
                weight += dist[ind[i * 4 + k]];
                new_weights[i * jointnum + j] += dist[ind[i * 4 + k]] *
                        weights[ind[i * 4 + k] * jointnum + j];
            }
            new_weights[i * jointnum + j] /= weight;
        }
    }

    // linear blend skinning for vertex [3]
    float *SMPL::lbs_for_custom_vertices(float *beta, float *theta, float *vertex) {
        auto d_shapeBlendShape = shapeBlendShape(beta);

        float *d_dist;
        cudaMalloc((void **) &d_dist, VERTEX_NUM * sizeof(float));
        int *d_ind;
        cudaMalloc((void **) &d_ind, 4 * sizeof(int));
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

        auto res = run(beta, theta, d_cur_weights, d_vertex, 1);
        cudaFree(d_cur_weights);
        cudaFree(d_vertex);

        return res;
    }
}