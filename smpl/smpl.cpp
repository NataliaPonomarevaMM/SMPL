#include <fstream>
// #include <experimental/filesystem>
#include <nlohmann/json.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>
#include "smpl.h"
#include "def.h"

namespace smpl {
    SMPL::SMPL() :
            d_poseBlendBasis(nullptr),
            d_shapeBlendBasis(nullptr),
            d_templateRestShape(nullptr),
            d_jointRegressor(nullptr),
            d_weights(nullptr),
            d_kinematicTree(nullptr)
    {
    }

    void SMPL::init(std::string &modelPath) {
        nlohmann::json model; // JSON object represents.
        std::ifstream file(modelPath);
        file >> model;

        // blender
        xt::xarray<float> shapeBlendBasis;
        xt::xarray<float> poseBlendBasis;
        xt::from_json(model["shape_blend_shapes"], shapeBlendBasis);
        xt::from_json(model["pose_blend_shapes"], poseBlendBasis);

        // regressor
        xt::xarray<float> templateRestShape;
        xt::xarray<float> jointRegressor;
        xt::from_json(model["vertices_template"], templateRestShape);
        xt::from_json(model["joint_regressor"], jointRegressor);

        // transformer
        xt::xarray<int64_t> kinematicTree;
        xt::from_json(model["kinematic_tree"], kinematicTree);

        // skinner
        xt::xarray<float> weights;
        xt::from_json(model["weights"], weights);

        loadToDevice(shapeBlendBasis.data(), poseBlendBasis.data(), templateRestShape.data(),
                jointRegressor.data(), kinematicTree.data(), weights.data());
    }
} // namespace smpl