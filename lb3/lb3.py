# plot.py
# pip install pandas matplotlib numpy   (или apt install python3-pandas python3-matplotlib python3-numpy)
# python3 plot.py            # читает results.csv, пишет *.png
# python3 plot.py my.csv     # свой csv

import sys

# --- ОБЯЗАТЕЛЬНО до import pyplot: неинтерактивный бэкенд ---
# Обходит GLib-GIO-ERROR: ...antialiasing в системах с GTK/GNOME
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(CSV)

# --- человекочитаемые подписи и маркеры ---
ALGOS = ["row_bands", "blocks", "cache_blocks", "cache_blocks_rowwise"]
LABELS = {
    "row_bands":            "Ленты по строкам",
    "blocks":               "Блочное",
    "cache_blocks":         "Кэш-блоки",
    "cache_blocks_rowwise": "Кэш-блоки + построчно",
}
MARKERS = {
    "row_bands":            "o",
    "blocks":               "s",
    "cache_blocks":         "^",
    "cache_blocks_rowwise": "D",
}
COLORS = {
    "row_bands":            "tab:blue",
    "blocks":               "tab:orange",
    "cache_blocks":         "tab:green",
    "cache_blocks_rowwise": "tab:red",
}

# оставляем только реально присутствующие в файле алгоритмы
present = [a for a in ALGOS if a in set(df["algo"])]

# ---------- 1. Время ----------
plt.figure(figsize=(9, 5.5))
for a in present:
    s = df[df["algo"] == a].sort_values("T")
    plt.plot(s["T"], s["t_geo"],
             marker=MARKERS[a], color=COLORS[a],
             linewidth=2, label=LABELS[a])
ref = df[(df["algo"] == "ref") & (df["T"] == 0)]
if not ref.empty:
    plt.axhline(ref["t_geo"].iloc[0], color="black",
                linestyle="--", alpha=0.6, label="ref (последовательный)")
plt.xlabel("Число потоков $T$")
plt.ylabel("Время, с (среднее геом.)")
plt.title("Время выполнения умножения матриц")
plt.xticks(sorted(df["T"].unique()))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("time.png", dpi=150)
plt.close()
print("saved time.png")

# ---------- 2. Ускорение ----------
plt.figure(figsize=(9, 5.5))
max_T = int(df["T"].max())
plt.plot([1, max_T], [1, max_T], "k--", alpha=0.4, label="Линейное $S=T$")

for a in present:
    s = df[df["algo"] == a].sort_values("T")
    t1 = s[s["T"] == 1]["t_geo"]
    if t1.empty:
        continue
    t1 = t1.iloc[0]
    S = t1 / s["t_geo"]
    plt.plot(s["T"], S,
             marker=MARKERS[a], color=COLORS[a],
             linewidth=2, label=LABELS[a])
    for T, Sv in zip(s["T"], S):
        plt.annotate(f"{Sv:.2f}", (T, Sv),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color=COLORS[a])

plt.xlabel("Число потоков $T$")
plt.ylabel("Ускорение $S(T)=t(1)/t(T)$")
plt.title("Ускорение относительно $T=1$")
plt.xticks(sorted(df["T"].unique()))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("speedup.png", dpi=150)
plt.close()
print("saved speedup.png")

# ---------- 3. Эффективность ----------
plt.figure(figsize=(9, 5.5))
for a in present:
    s = df[df["algo"] == a].sort_values("T")
    t1 = s[s["T"] == 1]["t_geo"]
    if t1.empty:
        continue
    t1 = t1.iloc[0]
    E = (t1 / s["t_geo"]) / s["T"]
    plt.plot(s["T"], E,
             marker=MARKERS[a], color=COLORS[a],
             linewidth=2, label=LABELS[a])
plt.axhline(1.0, color="black", linestyle=":", alpha=0.5)
plt.xlabel("Число потоков $T$")
plt.ylabel("Эффективность $E(T)=S(T)/T$")
plt.title("Эффективность распараллеливания")
plt.xticks(sorted(df["T"].unique()))
plt.ylim(0, 1.1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("efficiency.png", dpi=150)
plt.close()
print("saved efficiency.png")

# ---------- 4. Столбчатая диаграмма при максимальном T ----------
Tmax = int(df["T"].max())
sub = df[df["T"] == Tmax].set_index("algo")
xs, ys, cs, lbs = [], [], [], []
for a in present:
    if a in sub.index:
        xs.append(len(xs))
        ys.append(sub.loc[a, "t_geo"])
        cs.append(COLORS[a])
        lbs.append(LABELS[a])

plt.figure(figsize=(9, 5))
bars = plt.bar(xs, ys, color=cs)
for b, y in zip(bars, ys):
    plt.text(b.get_x() + b.get_width() / 2, y * 1.01,
             f"{y:.3f} c", ha="center", va="bottom", fontsize=9)
plt.xticks(xs, lbs, rotation=15)
plt.ylabel("Время, с (среднее геом.)")
plt.title(f"Время при $T={Tmax}$")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("bar_Tmax.png", dpi=150)
plt.close()
print("saved bar_Tmax.png")

# ---------- 5. Таблицы в консоль ----------
print("\n=== Время (геом.), с ===")
pivot_t = df[df["algo"].isin(present)].pivot_table(
    index="T", columns="algo", values="t_geo")
print(pivot_t.round(4))

print("\n=== Ускорение S(T) ===")
pivot_S = pivot_t.copy()
for col in pivot_S.columns:
    if 1 in pivot_S.index:
        pivot_S[col] = pivot_S.loc[1, col] / pivot_S[col]
print(pivot_S.round(3))

print("\n=== Эффективность E(T) ===")
pivot_E = pivot_S.copy()
for col in pivot_E.columns:
    pivot_E[col] = pivot_E[col] / pivot_E.index.to_series().values
print(pivot_E.round(3))