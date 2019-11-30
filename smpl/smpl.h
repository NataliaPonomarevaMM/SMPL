#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <gtest/gtest.h>
#include "def.h"

namespace smpl {
    class SMPL {
    protected:
        ///GPU
        float *d_poseBlendBasis; // Basis of the pose-dependent shape space, (6890, 3, 207).
        float *d_shapeBlendBasis; // Basis of the shape-dependent shape space, (6890, 3, 10).
        float *d_templateRestShape; // Template shape in rest pose, (6890, 3).
        float *d_jointRegressor; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        float *d_weights; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
        int64_t *d_kinematicTree; // Weights for linear blend skinning, (6890, 24).

        void loadToDevice(float *shapeBlendBasis, float *poseBlendBasis,
                                float *templateRestShape, float *jointRegressor,
                                int64_t *kinematicTree, float *weights);

        std::tuple<float *, float *, float *> poseBlendShape(float *theta);
        float *shapeBlendShape(float *beta);
        std::tuple<float *, float *> regressJoints(float *d_shapeBlendShape, float *d_poseBlendShape);
        float *transform(float *d_poseRotation, float *d_joints);
        float *skinning(float *d_transformation, float *d_custom_weights, float *d_vertices, float vertexnum);

        float *run(float *beta, float *theta, float *d_custom_weights, float *d_vertices = nullptr, float vertexnum = 0);
    public:
        // Constructor and Destructor
        SMPL();
        ~SMPL();

        // Load model data stored as JSON file into current application.
        void init(std::string &modelPath);
        // Run the model with a specific group of beta, theta.
        float *lbs_for_model(float *beta, float *theta);
        float *lbs_for_custom_vertices(float *beta, float *theta, float *vertices, int vertnum);
    };
} // namespace smpl
#endif // SMPL_H
