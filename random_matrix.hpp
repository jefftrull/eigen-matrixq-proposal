// shared methods for generating random sparse matrices

// Copyright (C) 2017 Jeffrey E. Trull <edaskel@att.net>

// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RANDOM_MATRIX_HPP
#define RANDOM_MATRIX_HPP

#include <random>
#include <Eigen/Sparse>

template<typename Float>
Eigen::SparseMatrix<Float>
RandomMatrixOfSize(std::default_random_engine & gen,
             Eigen::Index rows,
             Eigen::Index cols,
             float density) {
    using namespace Eigen;
    std::vector<Triplet<Float>> tripletList;
    std::uniform_real_distribution<Float> dist(0.0,1.0);
    for (Index i = 0; i < rows; i++) {
        for (Index j = 0; j < cols; j++) {
            // get a random number between 0 and 1
            if (dist(gen) < density) {
                // populate this entry
                tripletList.emplace_back(i, j, 10 * dist(gen));
            }
        }
    }
    SparseMatrix<Float> mat(rows, cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

template<typename Float>
Eigen::SparseMatrix<Float>
RandomMatrix(std::default_random_engine & gen,
             Eigen::Index max_dim,
             float density) {
    using namespace Eigen;
    std::uniform_int_distribution<Eigen::Index> dim(1, max_dim);
    Index x = dim(gen);
    Index y = dim(gen);
    return RandomMatrixOfSize<Float>(gen, x, y, density);
}

template<typename Float>
struct MatrixCache {
    Eigen::SparseMatrix<Float>
    getRandomMatrix(std::default_random_engine & gen,
                    Eigen::Index rows, Eigen::Index cols,
                    float density) {
        auto it = cache_.find(std::make_pair(rows, cols));
        if (it != cache_.end()) {
            return it->second;
        } else {
            it = cache_.emplace(std::make_pair(rows, cols),
                                RandomMatrixOfSize<Float>(gen, rows, cols, density)).first;
            return it->second;
        }
    }
private:
    std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::SparseMatrix<Float>> cache_;
};

#endif // RANDOM_MATRIX_HPP
