# SMPL

 C++ Implementation of SMPL - A Skinned Multi-Person Linear Model.
 
 ## Overview

This project implements a 3D human skinning model - SMPL: A Skinned
Multi-Person Linear Model with C++ and pure CUDA. The official SMPL model is available at http://smpl.is.tue.mpg.de.

For more details, see the [paper](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) 
published by Max Planck Institute for Intelligent Systems on SIGGRAPH ASIA 
2015.
 
## Prerequisites

1. [xtensor](https://github.com/QuantStack/xtensor): A C++ library meant for numerical analysis with multi-dimensional array expressions. 

2. [nlohmann_json](https://github.com/nlohmann/json): JSON for Modern C++.

    *Xtensor* loads data from and dumps data into JSONs through nlohmann's toolkit.
    
3. [CUDA](https://developer.nvidia.com/cuda-downloads): NVIDIA parallel computing platform.
 
## Usage

- Data preprocessing

  Download and extract the official data from http://smpl.is.tue.mpg.de/, you
  will get two files:
    
      basicModel_f_lbs_10_207_0_v1.0.0.pkl
      basicmodel_m_lbs_10_207_0_v1.0.0.pkl

  Run `preprocess.py` in [SMPL](preprocess.py) with Python 2 to convert the data into `.json` and `. npz` format. 

- Build and Run

  After installing all packages, you can compile SMPL from source:

      mkdir build
      cd build
      cmake ..
      make
      
- Pipeline

  Following the paper, we can generate a mesh with four steps.

  1. Generate pose blend shape and shape blend shape.

  2. Regress joints from vertices.

  3. Compute the transformation matrices for each joint.

  4. Linear Blend Skinning
 
 ## Reference

[1] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. "SMPL: a skinned multi-person linear model". ACM Trans. Graph. 34, 6, Article 248 (October 2015), 16 pages.
