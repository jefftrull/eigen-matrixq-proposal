// performance testing dense Q generation from sparse QR result
//
// Copyright (C) 2017 Jeffrey E. Trull <edaskel@att.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <benchmark/benchmark.h>

#include "random_matrix.hpp"

int main(int argc, char* argv[]) {
    using namespace Eigen;

    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: rt <density>\n";
        return 1;
    }

    // TODO use gbench mechanisms for these:

    float const density = std::atof(argv[1]);

    using Float = float;

    // create a random NxN sparse matrix
    using namespace Eigen;
    std::default_random_engine gen;
    MatrixCache<Float> matrices;    // cache to ensure we compare same matrices for each size

    // benchmark creating the Q matrix from the (sparse) Householder vectors
    // old method creates identity matrix, then multiplies

    benchmark::RegisterBenchmark(
        "GenerateQMatrix",
        [&](benchmark::State & state) {
            Index size = state.range(0);
            SparseMatrix<Float> mat = matrices.getRandomMatrix(gen, size, size, density);
            SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(mat);
            auto id_size = qr.matrixQ().rows();   // RHS size for multiply
            Matrix<Float, Dynamic, Dynamic> id =
                Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
            while (state.KeepRunning()) {
                Matrix<Float, Dynamic, Dynamic> q = qr.matrixQ() * id;
                benchmark::DoNotOptimize(q);
            }
        })->RangeMultiplier(10)->Range(10,1000);

    // new method uses specialization for dense identity matrix

    benchmark::RegisterBenchmark(
        "GenerateQMatrix-New",
        [&](benchmark::State & state) {
            Index size = state.range(0);
            SparseMatrix<Float> mat = matrices.getRandomMatrix(gen, size, size, density);
            SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(mat);
            auto id_size = qr.matrixQ().rows();   // RHS size for multiply
            while (state.KeepRunning()) {
                Matrix<Float, Dynamic, Dynamic> q = qr.matrixQ() * Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
                benchmark::DoNotOptimize(q);
            }
        })->RangeMultiplier(10)->Range(10,1000);

    benchmark::RunSpecifiedBenchmarks();

}
