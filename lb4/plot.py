import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Фикс: бэкенд ДО импорта pyplot; 'Agg' — только сохранение в файл
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv").sort_values("T")
par = df[df["T"] > 0]

plt.figure(figsize=(7, 4))
plt.plot(par["T"], par["time_geo"], "o-", label="CG OpenMP (геом. среднее)")
plt.axhline(df.loc[df["T"] == 0, "time_geo"].iloc[0], color="gray", ls="--", label="seq")
plt.xlabel("Число потоков T")
plt.ylabel("Время, с")
plt.xticks(par["T"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("time.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(par["T"], par["speedup"], "o-", label="S(T)")
plt.plot(par["T"], par["T"], "k--", label="идеал")
plt.xlabel("T")
plt.ylabel("Ускорение")
plt.xticks(par["T"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("speedup.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(par["T"], par["efficiency"], "o-")
plt.axhline(1.0, color="gray", ls="--")
plt.xlabel("T")
plt.ylabel("E(T)")
plt.xticks(par["T"])
plt.ylim(0, 1.1)
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency.png", dpi=150)
plt.close()