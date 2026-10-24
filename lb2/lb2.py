#!/usr/bin/env python3

import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "results.csv"
OUT_SPEEDUP = "speedup.png"
OUT_TIME = "time.png"

ALGOS = [
    ("t_row",   "t_row",   "row-bands"),
    ("t_col",   "t_col",   "col-bands"),
    ("t_block", "t_block", "blocks"),
]


def parse(path):
    data = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = int(row["threads"])
            data[T]["t_ref"].append(float(row["t_ref"]))
            for key, _, _ in ALGOS:
                data[T][key].append(float(row[key]))
            data[T]["_m"].append(int(row["m"]))
            data[T]["_n"].append(int(row["n"]))
            data[T]["_k"].append(int(row["k"]))
            data[T]["_p"].append(int(row["p"]))
            data[T]["_q"].append(int(row["q"]))
    return data


def mean(values):
    return float(np.mean(values)) if values else 0.0


def main():
    data = parse(CSV_FILE)
    if not data:
        raise SystemExit(f"Нет данных в {CSV_FILE}")

    threads = sorted(data.keys())

    avg = {}
    for T in threads:
        d = data[T]
        avg[T] = {
            "m": d["_m"][0], "n": d["_n"][0],
            "k": d["_k"][0], "p": d["_p"][0], "q": d["_q"][0],
            "reps": len(d["t_ref"]),
            "t_ref": mean(d["t_ref"]),
        }
        for key, _, _ in ALGOS:
            avg[T][key] = mean(d[key])

    print("=" * 72)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА (усреднение по повторениям)")
    print("=" * 72)
    for T in threads:
        a = avg[T]
        print()
        print(f"--- Число потоков T = {T} ---")
        print(f"  размер матрицы : m = {a['m']}, n = {a['n']}")
        print(f"  параметры      : k = {a['k']}, p = {a['p']}, q = {a['q']}")
        print(f"  повторов       : {a['reps']}")
        print(f"  время ref      : {a['t_ref']:.6f} s")
        print(f"  время row      : {a['t_row']:.6f} s")
        print(f"  время col      : {a['t_col']:.6f} s")
        print(f"  время block    : {a['t_block']:.6f} s")

    T1 = threads[0]
    base = avg[T1]
    speedup = {key: [] for key, _, _ in ALGOS}
    for T in threads:
        for key, _, _ in ALGOS:
            speedup[key].append(base[key] / avg[T][key])

    print()
    print("=" * 72)
    print(f"УСКОРЕНИЕ S(T) = t(T={T1}) / t(T)")
    print("=" * 72)
    header = f"{'T':>3} " + " ".join(f"{label:>12}" for _, _, label in ALGOS)
    print(header)
    for i, T in enumerate(threads):
        row = f"{T:>3} " + " ".join(f"{speedup[key][i]:>12.3f}" for key, _, _ in ALGOS)
        print(row)

    plt.figure(figsize=(8, 6))
    for key, _, label in ALGOS:
        plt.plot(threads, speedup[key], "o-", label=label, linewidth=2)
    plt.plot(threads, threads, "k--", linewidth=1, label="идеал S(T)=T")
    plt.xlabel("Число потоков T")
    plt.ylabel("Ускорение S(T)")
    plt.title("Масштабируемость GEMV (OpenMP)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.xticks(threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_SPEEDUP, dpi=150)
    print(f"\nСохранено: {OUT_SPEEDUP}")

    plt.figure(figsize=(8, 6))
    for key, _, label in ALGOS:
        times = [avg[T][key] for T in threads]
        plt.plot(threads, times, "s-", label=label, linewidth=2)
    plt.xlabel("Число потоков T")
    plt.ylabel("Время выполнения, с")
    plt.title("Время выполнения GEMV (OpenMP)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.xticks(threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_TIME, dpi=150)
    print(f"Сохранено: {OUT_TIME}")

main()