import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

CPP_CODE = r'''
#include <iostream>
#include <cmath>
#include <iomanip>
#include <omp.h>

using namespace std;

double func(double x) {
    return sqrt(x * (3.0 - x)) / (x + 1.0);
}

double simpson_integration(double a, double b, int n) {
    double h = (b - a) / n;
    double sum_odd = 0.0, sum_even = 0.0;
    #pragma omp parallel for reduction(+:sum_odd, sum_even)
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 1) sum_odd  += func(x);
        else            sum_even += func(x);
    }
    return (h / 3.0) * (func(a) + 4.0 * sum_odd + 2.0 * sum_even + func(b));
}

int main() {
    double a = 1.0, b = 1.2, eps = 1e-6;
    int n = 10;
    double I_prev = 0.0, I_curr = 0.0, error = 1.0;
    do {
        I_prev = I_curr;
        I_curr = simpson_integration(a, b, n);
        if (n > 10) error = fabs(I_curr - I_prev);
        n *= 2;
    } while (error > eps);

    const int MAX_THREADS = 8;
    const int N_BIG = 100000000;

    for (int p = 1; p <= MAX_THREADS; p++) {
        omp_set_num_threads(p);
        double t_start = omp_get_wtime();
        volatile double I_big = simpson_integration(a, b, N_BIG);
        double t_end = omp_get_wtime();
        cout << "RESULT " << p << " " << (t_end - t_start) << endl;
    }
    return 0;
}
'''

def compile_cpp():
    with open("simpson.cpp", "w") as f:
        f.write(CPP_CODE)
    r = subprocess.run(["g++", "-O2", "-fopenmp", "simpson.cpp", "-o", "simpson"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        return False
    return True

def run_and_parse():
    out = subprocess.run(["./simpson"], capture_output=True, text=True).stdout
    threads, times = [], []
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "RESULT":
            threads.append(int(parts[1]))
            times.append(float(parts[2]))
    return threads, times

def simulate(threads):
    T1 = 6.0
    S = 0.05
    overhead = 0.05
    return [T1 * (S + (1 - S) / p) + overhead * p for p in threads]

def plot_all(threads, times):
    threads = np.array(threads)
    times   = np.array(times)
    speedup = times[0] / times
    ideal   = threads.astype(float)
    efficiency = speedup / threads * 100

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Анализ масштабируемости OpenMP (8 потоков)",
                 fontsize=15, fontweight="bold")

    # --- Время ---
    ax[0].plot(threads, times, "o-", color="crimson", linewidth=2,
               markersize=8, label="Фактическое время")
    ax[0].plot(threads, times[0] / ideal, "k--", alpha=0.6,
               label="Идеальное (T(1)/p)")
    ax[0].set_xlabel("Число потоков p")
    ax[0].set_ylabel("Время T(p), с")
    ax[0].set_title("Время выполнения")
    ax[0].grid(True, linestyle=":", alpha=0.7)
    ax[0].set_xticks(threads)
    ax[0].legend()

    # --- Ускорение ---
    ax[1].plot(threads, speedup, "s-", color="royalblue", linewidth=2,
               markersize=8, label="Фактическое ускорение")
    ax[1].plot(threads, ideal, "k--", alpha=0.6, label="Идеальное (линейное)")
    ax[1].set_xlabel("Число потоков p")
    ax[1].set_ylabel("Ускорение S(p) = T(1)/T(p)")
    ax[1].set_title("Ускорение")
    ax[1].grid(True, linestyle=":", alpha=0.7)
    ax[1].set_xticks(threads)
    ax[1].legend()

    # --- Эффективность ---
    ax[2].plot(threads, efficiency, "^-", color="seagreen", linewidth=2,
               markersize=8, label="Эффективность")
    ax[2].axhline(100, color="k", linestyle="--", alpha=0.5, label="Идеал 100%")
    ax[2].set_xlabel("Число потоков p")
    ax[2].set_ylabel("Эффективность E(p), %")
    ax[2].set_title("Эффективность")
    ax[2].set_ylim(0, 110)
    ax[2].grid(True, linestyle=":", alpha=0.7)
    ax[2].set_xticks(threads)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig("speedup.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    if compile_cpp():
        print("Компиляция OK. Запуск реального замера...\n")
        threads, times = run_and_parse()
        print(f"{'p':>3} {'Time, s':>12} {'Speedup':>12} {'Efficiency, %':>15}")
        print("-" * 46)
        t1 = times[0]
        for p, t in zip(threads, times):
            s = t1 / t
            e = s / p * 100
            print(f"{p:>3} {t:>12.4f} {s:>12.4f} {e:>15.2f}")
        plot_all(threads, times)
    else:
        print("g++ не найден.\n")
