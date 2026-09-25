import subprocess
import os
import time
import matplotlib.pyplot as plt
import numpy as np

CPP_CODE = r'''
#include <iostream>
#include <cmath>
#include <omp.h>

double func(double x) {
    return sqrt(x * (3.0 - x)) / (x + 1.0);
}

double simpson_integration(double a, double b, int n) {
    double h = (b - a) / n;
    double sum_odd = 0.0, sum_even = 0.0;
    #pragma omp parallel for reduction(+:sum_odd, sum_even)
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 1) sum_odd += func(x);
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

    volatile double sink = 0.0;
    for (int k = 0; k < 200; k++) {
        sink += simpson_integration(a, b, 2000000);
    }
    std::cout << I_curr << std::endl;
    return 0;
}
'''

def compile_cpp():
    with open("simpson.cpp", "w") as f:
        f.write(CPP_CODE)
    result = subprocess.run(
        ["g++", "-O2", "-fopenmp", "simpson.cpp", "-o", "simpson"],
        capture_output=True, text=True
    )
    return result.returncode == 0

def measure_real(threads_list, repeats=3):
    times = []
    for t in threads_list:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(t)
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            subprocess.run(["./simpson"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
        times.append(best)
        print(f"Потоков: {t:3d} -> время: {best:.4f} с")
    return times

def measure_model(threads_list):
    T1 = 5.0
    S = 0.02
    overhead = 0.015
    return [T1 * (S + (1 - S) / p) + overhead * p for p in threads_list]

def plot(threads, times, mode):
    times = np.array(times)
    speedup = times[0] / times
    efficiency = speedup / np.array(threads) * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Анализ масштабируемости OpenMP-программы ({mode})",
                 fontsize=14, fontweight="bold")

    axes[0].plot(threads, times, "o-", color="crimson", linewidth=2)
    axes[0].set_xlabel("Число потоков")
    axes[0].set_ylabel("Время выполнения, с")
    axes[0].set_title("Время выполнения T(p)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_xticks(threads)

    axes[1].plot(threads, speedup, "s-", color="royalblue",
                 linewidth=2, label="Фактическое")
    axes[1].plot(threads, threads, "k--", alpha=0.5, label="Идеальное (линейное)")
    axes[1].set_xlabel("Число потоков")
    axes[1].set_ylabel("Ускорение S(p)")
    axes[1].set_title("Ускорение S(p) = T(1)/T(p)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    axes[1].set_xticks(threads)

    axes[2].plot(threads, efficiency, "^-", color="seagreen", linewidth=2)
    axes[2].axhline(100, color="k", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Число потоков")
    axes[2].set_ylabel("Эффективность, %")
    axes[2].set_title("Эффективность E(p) = S(p)/p · 100%")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].set_xticks(threads)

    plt.tight_layout()
    plt.savefig("speedup.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    threads_list = [2, 4, 16, 32, 64]

    if compile_cpp():
        print("Компиляция успешна. Выполняется реальный замер...\n")
        times = measure_real(threads_list)
        plot(threads_list, times, mode="реальные замеры")
    else:
        print("Компилятор g++ с OpenMP не найден.")
        print("Строится демонстрационная модель масштабируемости.\n")
        times = measure_model(threads_list)
        for t, tm in zip(threads_list, times):
            print(f"Потоков: {t:3d} -> время (модель): {tm:.4f} с")
        plot(threads_list, times, mode="демонстрационная модель")