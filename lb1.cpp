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
    double sum_odd = 0.0;
    double sum_even = 0.0;

    #pragma omp parallel for reduction(+:sum_odd, sum_even)
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 1) {
            sum_odd += func(x);
        } else {
            sum_even += func(x);
        }
    }

    return (h / 3.0) * (func(a) + 4.0 * sum_odd + 2.0 * sum_even + func(b));
}

int main() {
    double a = 1.0;
    double b = 1.2;
    double epsilon = 1e-6;

    int n = 10;
    double I_prev = 0.0;
    double I_curr = 0.0;
    double error = 1.0;
    int iterations = 0;

    cout << "Вычисление интеграла с точностью epsilon = " << epsilon << endl;
    cout << "Итерация\t N\t\t Интеграл\t Погрешность" << endl;

    do {
        I_prev = I_curr;
        I_curr = simpson_integration(a, b, n);

        if (iterations > 0) {
            error = fabs(I_curr - I_prev);
        } else {
            error = 1.0;
        }

        cout << iterations << "\t\t " << n << "\t\t "
             << fixed << setprecision(8) << I_curr << "\t "
             << scientific << error << endl;

        n *= 2;
        iterations++;

    } while (error > epsilon);

    cout << "Итоговый результат: " << fixed << setprecision(8) << I_curr << endl;
    cout << "Всего итераций: " << iterations << endl;
    cout << endl;

    const int MAX_THREADS = 12;
    const int N_BIG = 100000000;

    cout << "Анализ масштабируемости (N = " << N_BIG << ")" << endl;
    cout << "    Потоки      Время, с      Ускорение     Эффект., %" << endl;

    double t1 = 0.0;

    for (int p = 1; p <= MAX_THREADS; p++) {
        omp_set_num_threads(p);

        double t_start = omp_get_wtime();
        double I_big = simpson_integration(a, b, N_BIG);
        double t_end = omp_get_wtime();

        double t = t_end - t_start;
        if (p == 1) t1 = t;

        double speedup = t1 / t;
        double efficiency = speedup / p * 100.0;

        cout << setw(8)  << p
             << setw(15) << fixed << setprecision(4) << t
             << setw(15) << fixed << setprecision(4) << speedup
             << setw(15) << fixed << setprecision(2) << efficiency << endl;
    }

    return 0;
}