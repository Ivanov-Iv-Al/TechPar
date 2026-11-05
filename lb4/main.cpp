#include "cg.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <omp.h>

static double geo_mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double t : v) s += std::log(t);
    return std::exp(s / v.size());
}

static double rel_err(int n, const double* x_ref, const double* x) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { double d = x[i]-x_ref[i]; num += d*d; den += x_ref[i]*x_ref[i]; }
    return std::sqrt(num / den);
}

int main(int argc, char** argv) {
    int n    = (argc > 1) ? std::atoi(argv[1]) : 1000;
    int maxit = (argc > 2) ? std::atoi(argv[2]) : 2000;
    double tol = 1e-10;

    std::vector<double> A((size_t)n*n), b(n), x_ref(n), x(n);
    gen_spd(n, A.data(), 42);

    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += A[(size_t)i*n + j];
        b[i] = s;
    }
    for (int i = 0; i < n; ++i) x_ref[i] = 1.0;

    std::vector<double> ts;
    int it_seq = 0;
    for (int rep = 0; rep < 10; ++rep) {
        std::fill(x.begin(), x.end(), 0.0);
        double t0 = omp_get_wtime();
        it_seq = cg_seq(n, A.data(), b.data(), x.data(), maxit, tol);
        ts.push_back(omp_get_wtime() - t0);
    }
    double t_seq = geo_mean(ts);
    double err_seq = rel_err(n, x_ref.data(), x.data());
    std::printf("SEQ  n=%d it=%d t=%.6f s err=%.3e\n", n, it_seq, t_seq, err_seq);

    int thread_list[] = {1, 2, 4, 8};
    FILE* f = std::fopen("results.csv", "w");
    std::fprintf(f, "T,time_geo,time_arith,iter,rel_err,speedup,efficiency\n");
    std::fprintf(f, "0,%.6f,%.6f,%d,%.3e,1.0,1.0\n",
                 t_seq, std::accumulate(ts.begin(), ts.end(), 0.0)/ts.size(),
                 it_seq, err_seq);

    for (int T : thread_list) {
        std::vector<double> tt;
        int it = 0;
        for (int rep = 0; rep < 5; ++rep) {
            std::fill(x.begin(), x.end(), 0.0);
            double t0 = omp_get_wtime();
            it = cg_omp(n, A.data(), b.data(), x.data(), maxit, tol, T);
            tt.push_back(omp_get_wtime() - t0);
        }
        double tg = geo_mean(tt);
        double ta = std::accumulate(tt.begin(), tt.end(), 0.0) / tt.size();
        double e  = rel_err(n, x_ref.data(), x.data());
        std::fprintf(f, "%d,%.6f,%.6f,%d,%.3e,%.4f,%.4f\n",
                     T, tg, ta, it, e, t_seq/tg, (t_seq/tg)/T);
        std::printf("T=%d it=%d t_geo=%.6f t_arith=%.6f err=%.3e S=%.3f E=%.3f\n",
                    T, it, tg, ta, e, t_seq/tg, (t_seq/tg)/T);
    }
    std::fclose(f);
    return 0;
}