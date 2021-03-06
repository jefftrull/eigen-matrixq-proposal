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
    if (tripletList.empty()) {
        // try again
        return RandomMatrixOfSize<Float>(gen, rows, cols, density);
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
std::pair<Eigen::SparseMatrix<Float>, Eigen::SparseMatrix<Float>>
RandomMatrixProduct(std::default_random_engine & gen,
                    Eigen::Index max_dim,
                    float density) {
    using namespace Eigen;
    std::uniform_int_distribution<Eigen::Index> dim(1, max_dim);

    // to multiply two random matrices the column count of the first
    // must match the row count of the second
    Index r1   = dim(gen);
    Index c1r2 = dim(gen);
    Index c2   = dim(gen);

    return std::make_pair(RandomMatrixOfSize<Float>(gen, r1, c1r2, density),
                          RandomMatrixOfSize<Float>(gen, c1r2, c2, density));
}

template<typename Float>
struct MatrixCache {
    Eigen::SparseMatrix<Float>
    getRandomMatrix(std::default_random_engine & gen,
                    Eigen::Index rows, Eigen::Index cols,
                    float density) {
        auto it = cache_.find(std::make_tuple(rows, cols, density));
        if (it != cache_.end()) {
            return it->second;
        } else {
            it = cache_.emplace(std::make_tuple(rows, cols, density),
                                RandomMatrixOfSize<Float>(gen, rows, cols, density)).first;
            return it->second;
        }
    }
private:
    std::map<std::tuple<Eigen::Index, Eigen::Index, float>, Eigen::SparseMatrix<Float>> cache_;
};

#endif // RANDOM_MATRIX_HPP
