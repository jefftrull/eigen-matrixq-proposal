# Eigen MatrixQ Proposal

I am submitting a PR to Eigen to improve the performance of the `SparseQR::matrixQ()` method. Presently it works by constructing an identity matrix of the right size, then applying the Householder sequence to it to produce the final `Q`. That code is written for a generic right hand side matrix, but if we know it's an identity matrix there is an optimization available, described in [Golub and van Loan, 4th Ed](https://jhupbooks.press.jhu.edu/content/matrix-computations-0), that avoids some unnecessary calculations. This speedup is already present in the dense QR code but was presumably overlooked.

I also made an additional tweak to sparse Q application that gives a nice speedup for general operands.

This repo contains verification code (many random runs) and benchmarking code (with Google Benchmark) to demonstrate that the change is correct and produces improved performance.
