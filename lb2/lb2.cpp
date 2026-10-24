#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

static double now_sec(void) {
    return omp_get_wtime();
}

static void init_data(int m, int n, double *A, double *b) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * n + j] = (double)((i + j) % 7) * 0.5 + 0.1;
    for (int j = 0; j < n; ++j)
        b[j] = (double)(j % 5) * 0.3 + 0.2;
}

static int check(const double *ref, const double *res, int m) {
    double max_err = 0.0;
    for (int i = 0; i < m; ++i) {
        double e = fabs(ref[i] - res[i]);
        if (e > max_err) max_err = e;
    }
    return (max_err < 1e-9);
}

static void matvec_ref(int m, int n, const double *A, const double *b, double *c) {
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += A[(size_t)i * n + j] * b[j];
        c[i] = s;
    }
}

static void matvec_row_bands(int m, int n, const double *A, const double *b,
                             double *c, int k) {
    #pragma omp parallel for schedule(static)
    for (int band = 0; band < (m + k - 1) / k; ++band) {
        int i0 = band * k;
        int i1 = i0 + k; if (i1 > m) i1 = m;
        for (int i = i0; i < i1; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += A[(size_t)i * n + j] * b[j];
            c[i] = s;
        }
    }
}

static void matvec_col_bands(int m, int n, const double *A, const double *b,
                             double *c, int k) {
    int nbands = (n + k - 1) / k;

    #pragma omp parallel
    {
        double *local = (double *)calloc(m, sizeof(double));
        #pragma omp for schedule(static)
        for (int band = 0; band < nbands; ++band) {
            int j0 = band * k;
            int j1 = j0 + k; if (j1 > n) j1 = n;
            for (int i = 0; i < m; ++i) {
                double s = 0.0;
                for (int j = j0; j < j1; ++j)
                    s += A[(size_t)i * n + j] * b[j];
                local[i] += s;
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < m; ++i) c[i] += local[i];
        }
        free(local);
    }
}

static void matvec_blocks(int m, int n, const double *A, const double *b,
                          double *c, int p, int q) {
    int mb = (m + p - 1) / p;
    int nb = (n + q - 1) / q;
    int total = mb * nb;

    #pragma omp parallel
    {
        double *local = (double *)calloc(m, sizeof(double));
        #pragma omp for schedule(static)
        for (int t = 0; t < total; ++t) {
            int bi = t / nb;
            int bj = t % nb;
            int i0 = bi * p, i1 = i0 + p; if (i1 > m) i1 = m;
            int j0 = bj * q, j1 = j0 + q; if (j1 > n) j1 = n;
            for (int i = i0; i < i1; ++i) {
                double s = 0.0;
                for (int j = j0; j < j1; ++j)
                    s += A[(size_t)i * n + j] * b[j];
                local[i] += s;
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < m; ++i) c[i] += local[i];
        }
        free(local);
    }
}

static void run_one(const char *tag, int m, int n, const double *A, const double *b,
                    double *c, double *cr, int k, int p, int q,
                    int threads, int max_t, double *t_ref_out, double *t_alg_out,
                    int *ok_out, int algo) {
    omp_set_num_threads(threads);

    double t0 = now_sec();
    matvec_ref(m, n, A, b, cr);
    double t_ref = now_sec() - t0;

    for (int i = 0; i < m; ++i) c[i] = 0.0;
    t0 = now_sec();
    if (algo == 1) matvec_row_bands(m, n, A, b, c, k);
    else if (algo == 2) matvec_col_bands(m, n, A, b, c, k);
    else matvec_blocks(m, n, A, b, c, p, q);
    double t_alg = now_sec() - t0;

    int ok = check(cr, c, m);

    *t_ref_out = t_ref;
    *t_alg_out = t_alg;
    *ok_out = ok;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 8000;
    int n = (argc > 2) ? atoi(argv[2]) : 8000;
    int k = (argc > 3) ? atoi(argv[3]) : 64;
    int p = (argc > 4) ? atoi(argv[4]) : 64;
    int q = (argc > 5) ? atoi(argv[5]) : 64;

    int max_t = omp_get_max_threads();
    int thread_list[] = {1, 2, 4, 6, 8};
    int nthreads = (int)(sizeof(thread_list) / sizeof(thread_list[0]));

    double *A  = (double *)malloc((size_t)m * n * sizeof(double));
    double *b  = (double *)malloc((size_t)n * sizeof(double));
    double *c  = (double *)malloc((size_t)m * sizeof(double));
    double *cr = (double *)malloc((size_t)m * sizeof(double));

    init_data(m, n, A, b);

    FILE *csv = fopen("results.csv", "w");
    if (csv) {
        fprintf(csv, "m,n,k,p,q,threads,t_ref,t_row,t_col,t_block,ok_row,ok_col,ok_block\n");
    }

    printf("m=%d n=%d k=%d p=%d q=%d\n", m, n, k, p, q);

    for (int idx = 0; idx < nthreads; ++idx) {
        int T = thread_list[idx];
        if (T > max_t) T = max_t;

        double t_ref_r, t_ref_c, t_ref_b;
        double t_row, t_col, t_block;
        int ok_row, ok_col, ok_block;

        run_one("row",   m, n, A, b, c, cr, k, p, q, T, max_t, &t_ref_r, &t_row,   &ok_row,   1);
        run_one("col",   m, n, A, b, c, cr, k, p, q, T, max_t, &t_ref_c, &t_col,   &ok_col,   2);
        run_one("block", m, n, A, b, c, cr, k, p, q, T, max_t, &t_ref_b, &t_block, &ok_block, 3);

        double t_ref = t_ref_r;

        printf("\n Число потоков = %d \n", T);
        printf("  размер матрицы : m = %d, n = %d\n", m, n);
        printf("  параметры      : k = %d, p = %d, q = %d\n", k, p, q);
        printf("  время ref      : %.6f s\n", t_ref);
        printf("  время row      : %.6f s  ok=%d\n", t_row, ok_row);
        printf("  время col      : %.6f s  ok=%d\n", t_col, ok_col);
        printf("  время block    : %.6f s  ok=%d\n", t_block, ok_block);

        if (csv) {
            fprintf(csv, "%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d\n",
                    m, n, k, p, q, T,
                    t_ref, t_row, t_col, t_block,
                    ok_row, ok_col, ok_block);
        }
    }

    if (csv) fclose(csv);

    free(A); free(b); free(c); free(cr);
    return 0;
}