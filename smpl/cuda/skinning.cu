#include <cmath>
#include "../def.h"
#include "../smpl.h"

namespace smpl {
    namespace device {
        __global__ void
        Skinning(float *restShape, float *transformation, float *weights, int vertexnum, int jointnum,
                 float *vertices) {
            //restShape [vertexnum][3]
            //transformation [jointnum][4][4]
            //weights [vertexnum][jointnum]
            int j = threadIdx.x;

            float coeffs[16] = {0};
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 4; l++)
                    for (int t = 0; t < jointnum; t++)
                        coeffs[k * 4 + l] += weights[j * jointnum + t] * transformation[t * 16 + k * 4 + l];

            float homoW = coeffs[15];
            for (int t = 0; t < 3; t++)
                homoW += coeffs[12 + t] * restShape[j * 3 + t];
            for (int k = 0; k < 3; k++) {
                vertices[i * vertexnum * 3 + j * 3 + k] = coeffs[k * 4 + 3];
                for (int t = 0; t < 3; t++)
                    vertices[j * 3 + k] += coeffs[k * 4 + t] * restShape[j * 3 + t];
                vertices[j * 3 + k] /= homoW;
            }
        }
    }

    float *SMPL::skinning(float *d_restShape, float *d_transformation) {
        ///SKINNING
        float *d_vertices;
        cudaMalloc((void **) &d_vertices, VERTEX_NUM * 3 * sizeof(float));

        device::Skinning<<<1,VERTEX_NUM>>>(d_restShape, d_transformation, d_weights,
                VERTEX_NUM, JOINT_NUM, d_vertices);

        float *result_vertices = (float *)malloc(VERTEX_NUM * 3 * sizeof(float));
        cudaMemcpy(result_vertices, d_vertices, VERTEX_NUM * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vertices);
        return result_vertices;
    }
}