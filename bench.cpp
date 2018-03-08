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
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include <benchmark/benchmark.h>

#include "random_matrix.hpp"

int main(int argc, char* argv[]) {
    using namespace Eigen;

    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    using Float = float;

    // create a random NxN sparse matrix
    using namespace Eigen;
    std::default_random_engine gen;
    MatrixCache<Float> matrices;    // cache to ensure we compare same matrices for each size

    // benchmark creating the Q matrix from the (sparse) Householder vectors
    // via identity multiplication
    benchmark::RegisterBenchmark(
        "GenerateQMatrix",
        [&](benchmark::State & state) {
            Index size = state.range(0);
            SparseMatrix<Float> mat = matrices.getRandomMatrix(gen, size, size,
                                                               (float)(state.range(1))/100.);
            SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(mat);
            auto id_size = qr.matrixQ().rows();   // RHS size for multiply
            Matrix<Float, Dynamic, Dynamic> id =
                Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
            for (auto _ : state) {
                Matrix<Float, Dynamic, Dynamic> q =
                    qr.matrixQ() * Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
                benchmark::DoNotOptimize(q);
            }
        })->Ranges({{10, 1000}, {5, 20}});

    // now try the transposed versions of both
    benchmark::RegisterBenchmark(
        "GenerateQMatrix-Transpose",
        [&](benchmark::State & state) {
            Index size = state.range(0);
            SparseMatrix<Float> mat = matrices.getRandomMatrix(gen, size, size,
                                                               (float)(state.range(1))/100.);
            SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(mat);
            auto id_size = qr.matrixQ().rows();   // RHS size for multiply
            Matrix<Float, Dynamic, Dynamic> id =
                Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
            for (auto _ : state) {
                Matrix<Float, Dynamic, Dynamic> q =
                    qr.matrixQ().transpose() * Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
                benchmark::DoNotOptimize(q);
            }
        })->Ranges({{10, 1000}, {5, 20}});

    // benchmark multiplying the (implicit) Q matrix times a random dense matrix
    benchmark::RegisterBenchmark(
        "QMatrixProduct",
        [&](benchmark::State & state) {
            Index size = state.range(0);
            SparseMatrix<Float> mat = matrices.getRandomMatrix(gen, size, size,
                                                               (float)(state.range(1))/100.);
            SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(mat);
            auto rhs_size = qr.matrixQ().rows();   // RHS size for multiply
            Matrix<Float, Dynamic, Dynamic> rhs =
                Matrix<Float, Dynamic, Dynamic>::Random(rhs_size, rhs_size);
            for (auto _ : state) {
                Matrix<Float, Dynamic, Dynamic> q = qr.matrixQ() * rhs;
                benchmark::DoNotOptimize(q);
            }
        })->Ranges({{10, 1000}, {5, 20}});

    benchmark::RegisterBenchmark(
        "QMatrixProduct-Transpose",
        [&](benchmark::State & state) {
            Index size = state.range(0);
            SparseMatrix<Float> mat = matrices.getRandomMatrix(gen, size, size,
                                                               (float)(state.range(1))/100.);
            SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(mat);
            auto rhs_size = qr.matrixQ().rows();   // RHS size for multiply
            Matrix<Float, Dynamic, Dynamic> rhs =
                Matrix<Float, Dynamic, Dynamic>::Random(rhs_size, rhs_size);
            for (auto _ : state) {
                Matrix<Float, Dynamic, Dynamic> q = qr.matrixQ().transpose() * rhs;
                benchmark::DoNotOptimize(q);
            }
        })->Ranges({{10, 1000}, {5, 20}});

    benchmark::RunSpecifiedBenchmarks();

}
