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
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include <boost/iterator/counting_iterator.hpp>

#include "random_matrix.hpp"

int main(int argc, char* argv[]) {
    using namespace Eigen;

    if (argc < 3) {
        std::cerr << "Usage: verify <Matrix-dimension> <density>\n";
        return 1;
    }

    Index const size = std::atoi(argv[1]);
    float const density = std::atof(argv[2]);

    using Float = double;

    const int numTests = 1000000;
    std::default_random_engine gen;

    IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
    // let's test a bunch of matrices of different sizes
    for (int t = 0; t < numTests; t++) {
        // test sparse QR by performing it on a random matrix and then doing a solve

        // create a random sparse matrix of up to sizeXsize
        SparseMatrix<Float> sm = RandomMatrix<Float>(gen, size, density);

        // reject if it contains empty rows (SparseQR will not work!!)
        if (std::any_of(
                boost::counting_iterator<int>(0),
                boost::counting_iterator<int>(sm.rows()),
                [&sm](int row) {
                    // is this row absent in every column?
                    return std::all_of(
                        boost::counting_iterator<int>(0),
                        boost::counting_iterator<int>(sm.cols()),
                        [row,&sm](int col) {
                            // determine if this row is absent in the given column
                            bool found = false;
                            // InnerIterator is a Java-style iterator :-/
                            for (auto it = SparseMatrix<Float>::InnerIterator(sm, col); it; ++it) {
                                if (it.row() == row) {
                                    found = true;
                                    break;
                                }
                            }
                            return !found;
                        });
                })) {
            continue;
        }

        // Perform a sparse QR decomposition on it
        SparseQR<SparseMatrix<Float>, COLAMDOrdering<int>> qr(sm);
        if (qr.rank() == 0) {
            // this is just too degenerate to do anything with
            continue;
        }

        // now the dense version
        using MatrixDF = Matrix<Float, Dynamic, Dynamic>;
        MatrixDF dm(sm);
        auto denseqr = dm.colPivHouseholderQr();

        // an idea for an error threshold:
        // use epsilon times the number of operands involved, roughly
        // this is about 2e-5 for a 50x50 float matrix with 10% density
        // actually that was too low so I tweaked it... hacky :(
        Float error_threshold = (20*sm.rows()*sm.cols()*density)*std::numeric_limits<Float>::epsilon();

        // verify that we can recover the original matrix with Q*R*P'
        // The original SparseQR is a little weird here. It expects that the RHS of anything you
        // apply Q to will have the same number of rows as Q.  In the case of tall-and-thin matrices
        // that means you cannot multiply R on the left by its own Q!
        // IMO this is a bug as noted here:
        // https://listengine.tuxfamily.org/lists.tuxfamily.org/eigen/2017/01/msg00108.html
        // and I submitted a PR to fix it here:
        // https://bitbucket.org/eigen/eigen/pull-requests/367
        // For this code to work that fix must be in place

        MatrixDF sprecover = qr.matrixQ() * (MatrixDF(qr.matrixR().template triangularView<Upper>()) * qr.colsPermutation().transpose());
        if (((sprecover - MatrixDF(sm)).norm()/sprecover.norm()) > error_threshold) {
            std::cerr << "test " << t << ": could not recover original sparse matrix (norm " << (sprecover - MatrixDF(sm)).norm() << " vs threshold " << error_threshold << ")\n";
            std::cerr << "dense result had " << denseqr.matrixQ().rows() << "x" << denseqr.matrixQ().cols() << " Q matrix\n";
            std::cerr << "assigning " << qr.matrixQ().rows() << "x" << qr.matrixQ().cols() << " Q result to dense matrix\n";
            MatrixDF q = qr.matrixQ();
            std::cerr << "from Q =\n" << q.format(OctaveFmt) << "\n";
            std::cerr << "and R =\n" << MatrixDF(qr.matrixR().template triangularView<Upper>()).format(OctaveFmt) << "\n";
            std::cerr << "computed:\n" << sprecover.format(OctaveFmt) << "\n";
            std::cerr << "vs:\n" << MatrixDF(sm).format(OctaveFmt) << "\n";
            std::abort();
        }

        // Perform a dense QR decomposition on the same matrix
        // Try to recover with Q*R*P'
        MatrixDF denseR = denseqr.matrixR().template triangularView<Upper>();
        MatrixDF drecover = denseqr.matrixQ() * denseR * denseqr.colsPermutation().transpose();
        if (((drecover - dm).norm()/drecover.norm()) > error_threshold) {
            std::cerr << "could not recover original dense matrix (norm " << (drecover - dm).norm() << " vs threshold " << error_threshold << ")\n";
            std::cerr << "computed:\n" << drecover.format(OctaveFmt) << "\n";
            std::cerr << "vs:\n" << dm.format(OctaveFmt) << "\n";
            std::abort();
        }

        // try a solve
        if ((qr.rows() == qr.cols()) && (qr.rank() == qr.cols())) {
            // full rank -> invertible

            // Create a random dense matrix that the sparse Q (and hopefully the dense Q) can be applied to
            MatrixDF rhsmat = MatrixDF::Random(qr.rows(), qr.cols());

            // solve vs. this new RHS
            MatrixDF spresult = qr.solve(rhsmat);
            MatrixDF dresult  = denseqr.solve(rhsmat);
            // compare vs. dense result
            // This source: http://people.eecs.berkeley.edu/~demmel/cs267/lecture21/lecture21.html
            // suggests using the input's "condition number" to bound error checks
            JacobiSVD<MatrixXd> svd(dm);
            Float cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
            Float solve_error_threshold = 2 * cond * std::numeric_limits<Float>::epsilon();
            if (((spresult - dresult).norm()/spresult.norm()) > solve_error_threshold) {
                std::cerr << "solve produced different results (norm ratio " << ((spresult - dresult).norm()/spresult.norm()) << " vs limit " << solve_error_threshold << ")\n";
                std::cerr << "dense:\n" << dresult.format(OctaveFmt) << "\n";
                std::cerr << "sparse:\n" << spresult.format(OctaveFmt) << "\n";
                std::cerr << "for input matrix:\n" << dm.format(OctaveFmt) << "\n";
                std::cerr << "and rhs:\n" << rhsmat.format(OctaveFmt) << "\n";
                std::abort();
            }
        }

        // Verify that matrixQ() applied on the LHS of identity, and matrixQ assigned
        // to a dense matrix, are the same
        // We cannot simply compare the sparse and dense Q results because of pivoting
        MatrixDF id = MatrixDF::Identity(qr.rows(), qr.rows());
        MatrixDF q(qr.matrixQ());
        MatrixDF q_times_id = qr.matrixQ() * id;
        if ((q_times_id - q).norm() > error_threshold) {
            std::cerr << "matrixQ() * identity and matrixQ() converted to matrix differ!\n";
            std::cerr << "the former is:\n" << q_times_id.format(OctaveFmt) << "\nand the latter is:\n" << MatrixDF(q).format(OctaveFmt) << "\n";
            std::cerr << "the original matrix was:\n" << dm.format(OctaveFmt) << "\n";
            std::abort();
        }
        MatrixDF qt_times_id = qr.matrixQ().transpose() * id;
        // this does not work :(
        // MatrixDF qt(qr.matrixQ().transpose());
        if ((qt_times_id - q.transpose()).norm() > error_threshold) {
            std::cerr << "matrixQ().transpose() * identity and transposed matrixQ(), converted to matrix, differ!\n";
            std::cerr << "the former is:\n" << qt_times_id.format(OctaveFmt) << "\nand the latter is:\n" << MatrixDF(q.transpose()).format(OctaveFmt) << "\n";
            std::abort();
        }

        // Finally, check the operation of a "thin" Q, that is, applying it to a reduced identity
        // in order to get the first k columns
        if (qr.cols() >= 2) {
            auto k = q.cols() / 2;
            // two ways of forming the thin q
            MatrixDF thin_q = qr.matrixQ() * MatrixDF::Identity(q.cols(), k);
            if ((thin_q - q.leftCols(k)).norm() > error_threshold) {
                std::cerr << "thin Q formed from applying Q=\n" << q.format(OctaveFmt);
                std::cerr << "\nto " << k << " column identity gives wrong result:\n";
                std::cerr << thin_q.format(OctaveFmt) << "\n";
                std::abort();
            }
            MatrixDF thin_q_2 = qr.matrixQ() * MatrixDF::Identity(q.cols(), q.cols()).leftCols(k);
            if ((thin_q_2 - q.leftCols(k)).norm() > error_threshold) {
                std::cerr << "thin Q formed from applying Q=\n" << q.format(OctaveFmt);
                std::cerr << "\nto identity and taking the left " << k << " columns gives wrong result:\n";
                std::cerr << thin_q.format(OctaveFmt) << "\n";
                std::abort();
            }
            // now the transpose cases
            MatrixDF thin_q_t = qr.matrixQ().transpose() * MatrixDF::Identity(q.cols(), k);
            if ((thin_q_t - q.transpose().leftCols(k)).norm() > error_threshold) {
                std::cerr << "Q was " << q.format(OctaveFmt) << "\n";
                std::cerr << "thin Q formed from applying Q'=\n" << MatrixDF(q.transpose()).format(OctaveFmt);
                std::cerr << "\nto " << k << " column identity gives wrong result:\n";
                std::cerr << thin_q_t.format(OctaveFmt) << "\n";
                std::abort();
            }
            MatrixDF thin_q_t_2 = qr.matrixQ().transpose() * MatrixDF::Identity(q.cols(), q.cols()).leftCols(k);
            if ((thin_q_t_2 - q.transpose().leftCols(k)).norm() > error_threshold) {
                std::cerr << "thin Q formed from applying Q'=\n" << MatrixDF(q.transpose()).format(OctaveFmt);
                std::cerr << "\nto identity and taking the left " << k << " columns gives wrong result:\n";
                std::cerr << thin_q_t_2.format(OctaveFmt) << "\n";
                std::abort();
            }

        }
    }

    return 0;

}
