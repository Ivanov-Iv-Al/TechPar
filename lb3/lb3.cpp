#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

static void matmul_ref(int m, int l, int n,
                       const double *A, const double *B, double *C) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < l; ++p)
                s += A[(size_t)i * l + p] * B[(size_t)p * n + j];
            C[(size_t)i * n + j] = s;
        }
}

// ---------------------------------------------------------------
// 2. Ленточное разбиение по строкам шириной k
// ---------------------------------------------------------------
static void matmul_row_bands(int m, int l, int n,
                             const double *A, const double *B, double *C, int k) {
    int nbands = (m + k - 1) / k;
    #pragma omp parallel for schedule(static)
    for (int band = 0; band < nbands; ++band) {
        int i0 = band * k;
        int i1 = std::min(i0 + k, m);
        for (int i = i0; i < i1; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int p = 0; p < l; ++p)
                    s += A[(size_t)i * l + p] * B[(size_t)p * n + j];
                C[(size_t)i * n + j] = s;
            }
        }
    }
}

// ---------------------------------------------------------------
// 3. Блочное разбиение pb x rb для A, rb x qb для B
// ---------------------------------------------------------------
static void matmul_blocks(int m, int l, int n,
                          const double *A, const double *B, double *C,
                          int pb, int qb, int rb) {
    int MB = (m + pb - 1) / pb;
    int NB = (n + qb - 1) / qb;
    int KB = (l + rb - 1) / rb;
    int total = MB * NB;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < total; ++t) {
        int bi = t / NB;
        int bj = t % NB;
        int i0 = bi * pb, i1 = std::min(i0 + pb, m);
        int j0 = bj * qb, j1 = std::min(j0 + qb, n);

        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                C[(size_t)i * n + j] = 0.0;

        for (int kk = 0; kk < KB; ++kk) {
            int p0 = kk * rb;
            int p1 = std::min(p0 + rb, l);
            for (int i = i0; i < i1; ++i)
                for (int p = p0; p < p1; ++p) {
                    double a = A[(size_t)i * l + p];
                    for (int j = j0; j < j1; ++j)
                        C[(size_t)i * n + j] += a * B[(size_t)p * n + j];
                }
        }
    }
}

// ---------------------------------------------------------------
// 4. Кэш-блочное разбиение: блок block x block
// ---------------------------------------------------------------
static void matmul_cache_blocks(int m, int l, int n,
                                const double *A, const double *B, double *C,
                                int block) {
    int MB = (m + block - 1) / block;
    int NB = (n + block - 1) / block;
    int KB = (l + block - 1) / block;
    int total = MB * NB;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < total; ++t) {
        int bi = t / NB;
        int bj = t % NB;
        int i0 = bi * block, i1 = std::min(i0 + block, m);
        int j0 = bj * block, j1 = std::min(j0 + block, n);

        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                C[(size_t)i * n + j] = 0.0;

        for (int kk = 0; kk < KB; ++kk) {
            int p0 = kk * block, p1 = std::min(p0 + block, l);
            for (int i = i0; i < i1; ++i)
                for (int p = p0; p < p1; ++p) {
                    double a = A[(size_t)i * l + p];
                    for (int j = j0; j < j1; ++j)
                        C[(size_t)i * n + j] += a * B[(size_t)p * n + j];
                }
        }
    }
}

// ---------------------------------------------------------------
// 5. Кэш-блочное разбиение с построчным обходом блоков A и B
//    порядок i-p-j: обе матрицы читаются построчно
// ---------------------------------------------------------------
static void matmul_cache_blocks_rowwise(int m, int l, int n,
                                        const double *A, const double *B, double *C,
                                        int block) {
    int MB = (m + block - 1) / block;
    int NB = (n + block - 1) / block;
    int KB = (l + block - 1) / block;
    int total = MB * NB;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < total; ++t) {
        int bi = t / NB;
        int bj = t % NB;
        int i0 = bi * block, i1 = std::min(i0 + block, m);
        int j0 = bj * block, j1 = std::min(j0 + block, n);

        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                C[(size_t)i * n + j] = 0.0;

        for (int kk = 0; kk < KB; ++kk) {
            int p0 = kk * block, p1 = std::min(p0 + block, l);
            for (int i = i0; i < i1; ++i) {
                const double *Ai = A + (size_t)i * l;
                double *Ci = C + (size_t)i * n;
                for (int p = p0; p < p1; ++p) {
                    double a = Ai[p];
                    const double *Bp = B + (size_t)p * n;
                    for (int j = j0; j < j1; ++j)
                        Ci[j] += a * Bp[j];
                }
            }
        }
    }
}

// ---------------------------------------------------------------
// Единый интерфейс: один целочисленный параметр разбиения
// ---------------------------------------------------------------
typedef void (*AlgoFn)(int,int,int,const double*,const double*,double*,int);

static void algo_row_bands(int m, int l, int n,
                           const double *A, const double *B, double *C, int k) {
    matmul_row_bands(m, l, n, A, B, C, k);
}

static void algo_blocks(int m, int l, int n,
                        const double *A, const double *B, double *C, int pb) {
    // одинаковый размер блока по всем измерениям
    matmul_blocks(m, l, n, A, B, C, pb, pb, pb);
}

static void algo_cache_blocks(int m, int l, int n,
                              const double *A, const double *B, double *C, int block) {
    matmul_cache_blocks(m, l, n, A, B, C, block);
}

static void algo_cache_blocks_rowwise(int m, int l, int n,
                                      const double *A, const double *B, double *C, int block) {
    matmul_cache_blocks_rowwise(m, l, n, A, B, C, block);
}

// ---------------------------------------------------------------
// Вспомогательное
// ---------------------------------------------------------------
static double *alloc_aligned(size_t cnt) {
    size_t bytes = cnt * sizeof(double);
    bytes = (bytes + 63) & ~(size_t)63;
    return (double*)std::aligned_alloc(64, bytes);
}

static void fill_random(double *p, size_t nn, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < nn; ++i)
        p[i] = (double)rand() / RAND_MAX - 0.5;
}

static double max_err(const double *Cref, const double *C, size_t nn) {
    double e = 0.0;
    for (size_t i = 0; i < nn; ++i)
        e = std::max(e, std::fabs(Cref[i] - C[i]));
    return e;
}

static double geo_mean(const std::vector<double> &v) {
    double s = 0.0;
    for (double t : v) s += std::log(t);
    return std::exp(s / v.size());
}
static double arith_mean(const std::vector<double> &v) {
    double s = 0.0;
    for (double t : v) s += t;
    return s / v.size();
}

// ---------------------------------------------------------------
int main(int argc, char **argv) {
    int m = 2000, l = 2000, n = 2000;
    if (argc >= 4) { m = atoi(argv[1]); l = atoi(argv[2]); n = atoi(argv[3]); }

    // параметры разбиений
    int k_band = 64;   // ширина ленты
    int pb     = 128;  // блок для matmul_blocks (одинаковый по всем осям)
    int block  = 64;   // кэш-блок, 64*64*8 = 32 КБ -> L1

    size_t nnC = (size_t)m * n;
    size_t nnA = (size_t)m * l;
    size_t nnB = (size_t)l * n;

    double *A  = alloc_aligned(nnA);
    double *B  = alloc_aligned(nnB);
    double *C  = alloc_aligned(nnC);
    double *Cr = alloc_aligned(nnC);

    fill_random(A, nnA, 1);
    fill_random(B, nnB, 2);

    // эталон
    matmul_ref(m, l, n, A, B, Cr);

    printf("m=%d l=%d n=%d  ops=%.2e  (2*m*l*n)\n",
           m, l, n, 2.0 * m * l * n);

    int thread_list[] = {1, 2, 4, 6, 8};
    int max_t = omp_get_max_threads();

    FILE *csv = fopen("results.csv", "w");
    fprintf(csv, "T,algo,t_mean,t_geo,err\n");

    struct Entry { const char *name; AlgoFn fn; int arg; };
    Entry entries[] = {
        {"row_bands",            algo_row_bands,            k_band},
        {"blocks",               algo_blocks,               pb},
        {"cache_blocks",         algo_cache_blocks,         block},
        {"cache_blocks_rowwise", algo_cache_blocks_rowwise, block},
    };

    // --- эталон (последовательный), не зависит от T ---
    {
        std::vector<double> tt;
        for (int r = 0; r < 5; ++r) {
            double t0 = omp_get_wtime();
            matmul_ref(m, l, n, A, B, C);
            tt.push_back(omp_get_wtime() - t0);
        }
        double g = geo_mean(tt);
        double e = max_err(Cr, C, nnC);
        printf("%-22s geo=%8.4f  err=%.2e\n", "ref(seq)", g, e);
        fprintf(csv, "0,ref,%.6f,%.6f,%.2e\n", arith_mean(tt), g, e);
    }

    // --- для каждого T прогоняем все алгоритмы ---
    printf("\n%-22s %4s %10s %10s %10s %s\n",
           "algo", "T", "mean", "geo", "err", "status");
    for (int T : thread_list) {
        if (T > max_t) T = max_t;
        omp_set_num_threads(T);

        for (auto &e : entries) {
            std::vector<double> tt;
            for (int rep = 0; rep < 10; ++rep) {
                std::fill(C, C + nnC, 0.0);
                double t0 = omp_get_wtime();
                e.fn(m, l, n, A, B, C, e.arg);
                tt.push_back(omp_get_wtime() - t0);
            }
            double tmean = arith_mean(tt);
            double tgeo  = geo_mean(tt);
            double err   = max_err(Cr, C, nnC);
            const char *ok = (err < 1e-9) ? "OK" : "FAIL";
            printf("%-22s %4d %10.4f %10.4f %10.2e %s\n",
                   e.name, T, tmean, tgeo, err, ok);
            fprintf(csv, "%d,%s,%.6f,%.6f,%.2e\n",
                    T, e.name, tmean, tgeo, err);
        }
        printf("\n");
    }

    fclose(csv);
    free(A); free(B); free(C); free(Cr);
    return 0;
}