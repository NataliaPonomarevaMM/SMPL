#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <gtest/gtest.h>
#include "def.h"

namespace smpl {
    class SMPL : public ::testing::Test {
    protected:
        /// CPU
        int32_t *m__faceIndices; // Vertex indices of each face, (13776, 3)
        float *m__shapeBlendBasis; // Basis of the shape-dependent shape space, (6890, 3, 10).
        float *m__poseBlendBasis; // Basis of the pose-dependent shape space, (6890, 3, 207).
        float *m__templateRestShape; // Template shape in rest pose, (6890, 3).
        float *m__jointRegressor; // Joint coefficients of each vertices for regressing them to joint locations, (24, 6890).
        int64_t *m__kinematicTree; // Hierarchy relation between joints, the root is at the belly button, (2, 24).
        float *m__weights; // Weights for linear blend skinning, (6890, 24).

        ///GPU
        float *d_poseBlendBasis, *d_shapeBlendBasis, *d_templateRestShape, *d_jointRegressor, *d_weights;
        int64_t *d_kinematicTree;

        void loadToDevice();

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
        void init();
        // Run the model with a specific group of beta, theta.
        float *lbs_for_model(float *beta, float *theta);
        float *lbs_for_custom_vertices(float *beta, float *theta, float *vertex);
    };
} // namespace smpl
#endif // SMPL_H
