#include "cg.h"
#include <cmath>
#include <cstring>
#include <random>
#include <omp.h>

void matvec(int n, const double* A, const double* x, double* y) {
    for (int i = 0; i < n; ++i) {
        const double* Ai = A + (size_t)i * n;
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += Ai[j] * x[j];
        y[i] = s;
    }
}

double dot(int n, const double* a, const double* b) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

int cg_seq(int n, const double* A, const double* b, double* x,
           int maxit, double tol) {
    double* r  = new double[n];
    double* p  = new double[n];
    double* Ap = new double[n];

    std::memset(x, 0, sizeof(double) * n);
    std::memcpy(r, b, sizeof(double) * n);   // r0 = b - A*x0 = b
    std::memcpy(p, r, sizeof(double) * n);   // p0 = r0

    double rr = dot(n, r, r);
    double bnorm = std::sqrt(dot(n, b, b));
    int it = 0;

    for (; it < maxit; ++it) {
        matvec(n, A, p, Ap);
        double pAp = dot(n, p, Ap);
        double alpha = rr / pAp;

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        double rr_new = dot(n, r, r);
        if (std::sqrt(rr_new) / bnorm < tol) { ++it; break; }

        double beta = rr_new / rr;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }
    delete[] r; delete[] p; delete[] Ap;
    return it;
}

int cg_omp(int n, const double* A, const double* b, double* x,
           int maxit, double tol, int threads) {
    omp_set_num_threads(threads);

    double* r  = new double[n];
    double* p  = new double[n];
    double* Ap = new double[n];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) { x[i] = 0.0; r[i] = b[i]; p[i] = b[i]; }

    double rr = 0.0, bnorm = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:rr,bnorm)
    for (int i = 0; i < n; ++i) { rr += r[i]*r[i]; bnorm += b[i]*b[i]; }
    bnorm = std::sqrt(bnorm);

    int it = 0;
    for (; it < maxit; ++it) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double* Ai = A + (size_t)i * n;
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += Ai[j] * p[j];
            Ap[i] = s;
        }

        double pAp = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:pAp)
        for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        double alpha = rr / pAp;

        double rr_new = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:rr_new)
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rr_new += r[i] * r[i];
        }
        if (std::sqrt(rr_new) / bnorm < tol) { ++it; break; }

        double beta = rr_new / rr;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }
    delete[] r; delete[] p; delete[] Ap;
    return it;
}

//SPD матрица
void gen_spd(int n, double* A, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    double* M = new double[(size_t)n * n];
    for (size_t k = 0; k < (size_t)n * n; ++k) M[k] = d(gen);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) s += M[(size_t)k*n + i] * M[(size_t)k*n + j];
            A[(size_t)i*n + j] = s + (i == j ? (double)n : 0.0);
        }
    }
    delete[] M;
}