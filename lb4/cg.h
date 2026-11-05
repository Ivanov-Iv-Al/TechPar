#pragma once
#include <cstddef>

void matvec(int n, const double* A, const double* x, double* y);

double dot(int n, const double* a, const double* b);

int cg_seq(int n, const double* A, const double* b, double* x,
           int maxit, double tol);

int cg_omp(int n, const double* A, const double* b, double* x,
           int maxit, double tol, int threads);

void gen_spd(int n, double* A, unsigned seed);