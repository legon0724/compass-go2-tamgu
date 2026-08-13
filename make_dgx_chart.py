from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sizes = np.array([512, 1024, 2048, 4096, 8192])
fp32 = np.array([0.04, 0.13, 0.82, 6.31, 50.29])
tf32 = np.array([0.04, 0.09, 0.53, 4.03, 32.02])
int8 = np.array([0.03, 0.05, 0.18, 1.25, 9.80])

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "font.size": 11,
})

x = np.arange(len(sizes))
width = 0.24
fig, ax = plt.subplots(figsize=(10.6, 5.8), dpi=180)

bars = [
    ax.bar(x - width, fp32, width, label="FP32", color="#355C9A"),
    ax.bar(x, tf32, width, label="TF32", color="#55A868"),
    ax.bar(x + width, int8, width, label="INT8", color="#E68A3F"),
]

ax.set_yscale("log")
ax.set_xlabel("행렬 크기 N")
ax.set_ylabel("행렬곱 실행시간 (ms, 로그 눈금)")
ax.set_xticks(x, [str(v) for v in sizes])
ax.grid(axis="y", which="both", linestyle="--", alpha=0.28)
ax.legend(ncol=3, loc="upper left")

for group in bars:
    for bar in group:
        value = bar.get_height()
        ax.annotate(
            f"{value:g}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = Path(__file__).with_name("dgx_precision_runtime_chart.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
