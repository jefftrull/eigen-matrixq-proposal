// Verifying that my revised sparse Q conversion code is good
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

#include "random_matrix.hpp"

int main(int argc, char* argv[]) {
    using namespace Eigen;

    if (argc < 3) {
        std::cerr << "Usage: verify <Matrix-dimension> <density>\n";
        return 1;
    }

    Index const size = std::atoi(argv[1]);
    float const density = std::atof(argv[2]);

    using Float = float;

    const int numTests = 1000000;
    std::default_random_engine gen;

    // let's test a bunch of matrices of different sizes
    for (int t = 0; t < numTests; t++) {
        // TODO remove
        Eigen::base_running = false;
        Eigen::specialized_running = false;

        // create a random sparse matrix of up to sizeXsize
        SparseMatrix<Float> sm = RandomMatrix<Float>(gen, size, density);
        SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(sm);
        auto id_size = qr.matrixQ().rows();
        // old path will create identity matrix and multiply by it

        // invoke old (strictly a multiplication) code by creating a dense matrix that happens
        // to be identity
        Matrix<Float, Dynamic, Dynamic> id =
            Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
        Matrix<Float, Dynamic, Dynamic> qold = qr.matrixQ()*id;

        // new path uses specialization for identity type
        Matrix<Float, Dynamic, Dynamic> qnew = qr.matrixQ() * Matrix<Float, Dynamic, Dynamic>::Identity(id_size, id_size);
        if (!Eigen::base_running) {
            std::cerr << "did not use base\n";
            return 1;
        }
        if (!Eigen::specialized_running) {
            std::cerr << "did not use specialization\n";
            return 1;
        }
        if (qnew != qold) {
            std::cerr << "FAIL!\n";
            IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
            Matrix<Float, Dynamic, Dynamic> dm(sm);
            std::cerr << "A=\n" << Matrix<Float, Dynamic, Dynamic>(dm).format(OctaveFmt) << "\n";
            Matrix<Float, Dynamic, Dynamic> dold(qold);
            std::cerr << "Q(old)=\n" << Matrix<Float, Dynamic, Dynamic>(dold).format(OctaveFmt) << "\n";
            Matrix<Float, Dynamic, Dynamic> dnew(qnew);
            std::cerr << "Q(new)=\n" << Matrix<Float, Dynamic, Dynamic>(dnew).format(OctaveFmt) << "\n";

            return 1;
        }
    }

    return 0;

}
