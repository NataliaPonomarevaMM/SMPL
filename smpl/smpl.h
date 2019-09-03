#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <gtest/gtest.h>
#include "TorchEx.h"

namespace smpl {
    class SMPL final : public ::testing::Test{
    private:
        torch::Device m__device; // Torch device to run the module, could be CPUs or GPUs.

        std::string m__modelPath; // Path to the JSON model file.
        std::string m__vertPath; // Path to store the mesh OBJ file.
        nlohmann::json m__model; // JSON object represents.

        torch::Tensor m__faceIndices; // Vertex indices of each face, (13776, 3)
        torch::Tensor m__shapeBlendBasis; // Basis of the shape-dependent shape space, (6890, 3, 10).
        torch::Tensor m__poseBlendBasis; // Basis of the pose-dependent shape space, (6890, 3, 207).
        torch::Tensor m__templateRestShape; // Template shape in rest pose, (6890, 3).
        torch::Tensor m__jointRegressor; // Joint coefficients of each vertices for
                                        // regressing them to joint locations, (24, 6890).
        torch::Tensor m__kinematicTree; // Hierarchy relation between joints,
                                        // the root is at the belly button, (2, 24).
        torch::Tensor m__weights; // Weights for linear blend skinning, (6890, 24).

        torch::Tensor m__result_vertices;// result of launch

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        blendShape(torch::Tensor &beta, torch::Tensor &theta);
        std::tuple<torch::Tensor, torch::Tensor> regressJoints(torch::Tensor &beta, torch::Tensor &theta);
        torch::Tensor transform(torch::Tensor &poseRotation, torch::Tensor &joints);
        torch::Tensor skinning(torch::Tensor &restShape, torch::Tensor &transformation);

        torch::Tensor rodrigues(torch::Tensor &theta);
        torch::Tensor localTransform(torch::Tensor &poseRotHomo);
        torch::Tensor globalTransform(torch::Tensor &localTransformations);
        torch::Tensor cart2homo(torch::Tensor &cart);
        torch::Tensor homo2cart(torch::Tensor &homo);
    public:
        // Constructor and Destructor
        SMPL();
        SMPL(std::string &modelPath, std::string &vertPath, torch::Device &device);
        ~SMPL() = default;

        // Setter and Getter
        void setDevice(const torch::Device &device);
        void setModelPath(const std::string &modelPath);
        void setVertPath(const std::string &vertexPath);

        // Modeling
        // Load model data stored as JSON file into current application.
        void init();
        // Run the model with a specific group of beta, theta, and translation.
        void launch(torch::Tensor &beta, torch::Tensor &theta);
        // Export the deformed mesh to OBJ file.
        void out(int64_t index);
    };
} // namespace smpl
#endif // SMPL_H
