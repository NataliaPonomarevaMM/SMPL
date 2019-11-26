#include <fstream>
// #include <experimental/filesystem>
#include <nlohmann/json.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>
#include "../../include/smpl/exception.h"
#include "../../include/smpl/smpl.h"
#include "../../include/def.h"

namespace smpl {
    SMPL::SMPL() :
            m__modelPath(""),
            m__vertPath(""),
            m__faceIndices(nullptr),
            m__shapeBlendBasis(nullptr),
            m__poseBlendBasis(nullptr),
            m__templateRestShape(nullptr),
            m__jointRegressor(nullptr),
            m__kinematicTree(nullptr),
            m__weights(nullptr),
            m__result_vertices(nullptr),
            d_poseBlendBasis(nullptr),
            d_shapeBlendBasis(nullptr),
            d_templateRestShape(nullptr),
            d_jointRegressor(nullptr),
            d_weights(nullptr),
            d_kinematicTree(nullptr)
    {
    }

    SMPL::SMPL(std::string &modelPath, std::string &vertPath) {
        m__modelPath = modelPath;
        m__vertPath = vertPath;
    }

    void SMPL::init() {
        nlohmann::json model; // JSON object represents.
        std::ifstream file(m__modelPath);
        file >> model;

        // face indices
        xt::xarray<int32_t> faceIndices;
        xt::from_json(model["face_indices"], faceIndices);
        m__faceIndices = faceIndices.data();

        // blender
        xt::xarray<float> shapeBlendBasis;
        xt::xarray<float> poseBlendBasis;
        xt::from_json(model["shape_blend_shapes"], shapeBlendBasis);
        xt::from_json(model["pose_blend_shapes"], poseBlendBasis);
        m__shapeBlendBasis = shapeBlendBasis.data();// (6890, 3, 10)
        m__poseBlendBasis = poseBlendBasis.data();// (6890, 3, 207)

        // regressor
        xt::xarray<float> templateRestShape;
        xt::xarray<float> jointRegressor;
        xt::from_json(model["vertices_template"], templateRestShape);
        xt::from_json(model["joint_regressor"], jointRegressor);
        m__templateRestShape = templateRestShape.data();// (6890, 3)
        m__jointRegressor = jointRegressor.data();// (24, 6890)

        // transformer
        xt::xarray<int64_t> kinematicTree;
        xt::from_json(model["kinematic_tree"], kinematicTree);
        m__kinematicTree = kinematicTree.data();// (2, 24)

        // skinner
        xt::xarray<float> weights;
        xt::from_json(model["weights"], weights);
        m__weights = weights.data();// (6890, 24)

        loadToDevice();
    }

/**out
 *      @index: - size_t -
 *          A mesh in the batch to be exported.
 */
    void SMPL::out(int64_t ind) {
        std::ofstream file(m__vertPath);

        for (int64_t i = 0; i < VERTEX_NUM; i++)
            file << 'v' << ' ' << m__result_vertices[ind * VERTEX_NUM * 3 + i * 3] << ' '
                               << m__result_vertices[ind * VERTEX_NUM * 3 + i * 3 + 1] << ' '
                               << m__result_vertices[ind * VERTEX_NUM * 3 + i * 3 + 2] << '\n';

        for (int64_t i = 0; i < FACE_INDEX_NUM; i++)
            file << 'f' << ' ' << m__faceIndices[i * 3] << ' ' << m__faceIndices[i * 3 + 1] << ' ' << m__faceIndices[i * 3 + 2] << '\n';
    }
} // namespace smpl