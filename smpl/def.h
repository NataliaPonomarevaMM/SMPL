#ifndef DEF_H
#define DEF_H

#ifndef VERTEX_NUM
#define VERTEX_NUM smpl::vertex_num
#endif // VERTEX_NUM

#ifndef JOINT_NUM
#define JOINT_NUM smpl::joint_num
#endif // JOINT_NUM

#ifndef SHAPE_BASIS_DIM
#define SHAPE_BASIS_DIM smpl::shape_basis_dim
#endif // SHAPE_BASIS_DIM

#ifndef POSE_BASIS_DIM
#define POSE_BASIS_DIM smpl::pose_basis_dim
#endif // POSE_BASIS_DIM

#ifndef FACE_INDEX_NUM
#define FACE_INDEX_NUM smpl::face_index_num
#endif // FACE_INDEX_NUM

#include <cstdlib>

namespace smpl {
    extern int64_t vertex_num;// 6890
    extern const int64_t joint_num;// 24
    extern const int64_t shape_basis_dim;// 10
    extern const int64_t pose_basis_dim;// 207
    extern const int64_t face_index_num;// 13776
} // namespace smpl
#endif // DEF_H