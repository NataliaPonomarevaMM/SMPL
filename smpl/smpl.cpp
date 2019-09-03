#include <fstream>
#include <experimental/filesystem>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xjson.hpp>
#include "exception.h"
#include "smpl.h"
#include "../def.h"

namespace smpl {
    SMPL::SMPL() :
            m__device(torch::kCPU),
            m__modelPath(),
            m__vertPath(),
            m__faceIndices(),
            m__shapeBlendBasis(),
            m__poseBlendBasis(),
            m__templateRestShape(),
            m__jointRegressor(),
            m__kinematicTree(),
            m__weights(),
            m__model()
    {
    }

    SMPL::SMPL(std::string &modelPath, std::string &vertPath, torch::Device &device) :
            m__device(torch::kCPU),
            m__model()
    {
        if (device.has_index())
            m__device = device;
        else
            throw smpl_error("SMPL", "Failed to fetch device index!");

        std::experimental::filesystem::path path(modelPath);
        if (std::experimental::filesystem::exists(path)) {
            m__modelPath = modelPath;
            m__vertPath = vertPath;
        }
        else
            throw smpl_error("SMPL", "Failed to initialize model path!");
    }

    void SMPL::setDevice(const torch::Device &device) {
        if (device.has_index())
            m__device = device;
        else
            throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    void SMPL::setModelPath(const std::string &modelPath) {
        std::experimental::filesystem::path path(modelPath);
        if (std::experimental::filesystem::exists(path))
            m__modelPath = modelPath;
        else
            throw smpl_error("SMPL", "Failed to initialize model path!");
    }

    void SMPL::setVertPath(const std::string &vertexPath) {
        m__vertPath = vertexPath;
    }

    void SMPL::init() {
        std::experimental::filesystem::path path(m__modelPath);
        if (!std::experimental::filesystem::exists(path))
            throw smpl_error("SMPL", "Cannot initialize a SMPL model!");

        std::ifstream file(path);
        file >> m__model;

        // face indices
        xt::xarray<int32_t> faceIndices;
        xt::from_json(m__model["face_indices"], faceIndices);
        m__faceIndices = torch::from_blob(faceIndices.data(), {FACE_INDEX_NUM, 3}, torch::kInt32)
                .clone().to(m__device);

        // blender
        xt::xarray<float> shapeBlendBasis;
        xt::xarray<float> poseBlendBasis;
        xt::from_json(m__model["shape_blend_shapes"], shapeBlendBasis);
        xt::from_json(m__model["pose_blend_shapes"], poseBlendBasis);
        m__shapeBlendBasis = torch::from_blob(shapeBlendBasis.data(),
                                              {VERTEX_NUM, 3, SHAPE_BASIS_DIM}).to(m__device);// (6890, 3, 10)
        m__poseBlendBasis = torch::from_blob(poseBlendBasis.data(),
                                             {VERTEX_NUM, 3, POSE_BASIS_DIM}).to(m__device);// (6890, 3, 207)

        // regressor
        xt::xarray<float> templateRestShape;
        xt::xarray<float> jointRegressor;
        xt::from_json(m__model["vertices_template"], templateRestShape);
        xt::from_json(m__model["joint_regressor"], jointRegressor);
        m__templateRestShape = torch::from_blob(templateRestShape.data(),
                                                {VERTEX_NUM, 3}).to(m__device);// (6890, 3)
        m__jointRegressor = torch::from_blob(jointRegressor.data(),
                                             {JOINT_NUM, VERTEX_NUM}).to(m__device);// (24, 6890)

        // transformer
        xt::xarray<int64_t> kinematicTree;
        xt::from_json(m__model["kinematic_tree"], kinematicTree);
        m__kinematicTree = torch::from_blob(kinematicTree.data(),
                                            {2, JOINT_NUM}, torch::kInt64).to(m__device);// (2, 24)

        // skinner
        xt::xarray<float> weights;
        xt::from_json(m__model["weights"], weights);
        m__weights = torch::from_blob(weights.data(),
                                      {VERTEX_NUM, JOINT_NUM}).to(m__device);// (6890, 24)
    }

    void SMPL::launch(torch::Tensor &beta, torch::Tensor &theta) {
        if (beta.sizes() != torch::IntArrayRef({BATCH_SIZE, SHAPE_BASIS_DIM})
            && theta.sizes() != torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3}))
            throw smpl_error("SMPL", "Cannot launch a SMPL model!");

        auto [poseRotation, restPoseRotation, poseBlendShape, shapeBlendShape] = blendShape(beta, theta);
        auto [restShape, joints] = regressJoints(shapeBlendShape, poseBlendShape);
        auto transformation = transform(poseRotation, joints);
        m__result_vertices = skinning(restShape, transformation);
    }

/**out
 *      @index: - size_t -
 *          A mesh in the batch to be exported.
 */
    void SMPL::out(int64_t index) {
        if (m__result_vertices.sizes() != torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})
            || m__faceIndices.sizes() != torch::IntArrayRef({FACE_INDEX_NUM, 3}))
            throw smpl_error("SMPL", "Cannot export the deformed mesh!");

        std::ofstream file(m__vertPath);
        torch::Tensor slice_ = TorchEx::indexing(m__result_vertices, torch::IntList({index}));// (6890, 3)
        xt::xarray<float> slice = xt::adapt(
                (float *)slice_.to(torch::kCPU).data_ptr(),
                xt::xarray<float>::shape_type({(const size_t)VERTEX_NUM, 3})
        );

        xt::xarray<int32_t> faceIndices;
        faceIndices = xt::adapt(
                (int32_t *)m__faceIndices.to(torch::kCPU).data_ptr(),
                xt::xarray<int32_t>::shape_type(
                        {(const size_t)FACE_INDEX_NUM, 3})
        );

        for (int64_t i = 0; i < VERTEX_NUM; i++)
            file << 'v' << ' ' << slice(i, 0) << ' ' << slice(i, 1) << ' ' << slice(i, 2) << '\n';

        for (int64_t i = 0; i < FACE_INDEX_NUM; i++)
            file << 'f' << ' ' << faceIndices(i, 0) << ' ' << faceIndices(i, 1) << ' ' << faceIndices(i, 2) << '\n';
    }
} // namespace smpl