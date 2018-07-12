// performance testing dense Q generation from sparse QR result
// this is a port of ggael's sparse QR benchmark code to gbench
//
// Copyright (C) 2018 Jeffrey E. Trull <edaskel@att.net>
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
#include <unsupported/Eigen/SparseExtra>

#include <benchmark/benchmark.h>

int main(int argc, char* argv[]) {
    using namespace Eigen;

    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    if (argc < 1) {
        std::cerr << "please supply a MatrixMarket input file\n";
        return 1;
    }

    using Float = double;
    SparseMatrix<Float> sA;
    SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr;

    loadMarket( sA, argv[1] );

    // benchmark things a la ggael
    benchmark::RegisterBenchmark(
        "QR facto",
        [&](benchmark::State & state) {
            for (auto _ : state) {
                qr.compute(sA);
            }
        });

    VectorXd b = sA * VectorXd::Random(sA.cols());

    benchmark::RegisterBenchmark(
        "QR solve",
        [&](benchmark::State & state) {
            VectorXd x1;
            for (auto _ : state) {
                x1 = qr.solve(b);
                benchmark::DoNotOptimize(x1);
            }
        });

    benchmark::RegisterBenchmark(
        "Dense Q",
        [&](benchmark::State & state) {
            MatrixXd Q_dense;
            for (auto _ : state) {
                Q_dense = qr.matrixQ() * MatrixXd::Identity(sA.rows(),sA.rows());
                benchmark::DoNotOptimize(Q_dense);
            }
        });

    benchmark::RegisterBenchmark(
        "Q*b",
        [&](benchmark::State & state) {
            VectorXd z(sA.rows());
            for (auto _ : state) {
                z = qr.matrixQ() * b;
                benchmark::DoNotOptimize(z);
            }
        });

    benchmark::RegisterBenchmark(
        "Q*B_",
        [&](benchmark::State & state) {
            Index depth = state.range(0);
            MatrixXd Z(sA.rows(),depth), B(sA.rows(),depth);
            B.setRandom();
            for (auto _ : state) {
                Z = qr.matrixQ() * B;
                benchmark::DoNotOptimize(Z);
            }
        })->RangeMultiplier(2)->Range(5, 1000);    // "depth"


    benchmark::RunSpecifiedBenchmarks();

}
