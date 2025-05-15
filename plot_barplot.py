import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy as np
import pandas as pd


plt.rcParams.update({
    'axes.facecolor': 'white',        # 白色背景
    'grid.color': 'gray',             # 网格线颜色
    'grid.linestyle': '--',           # 虚线
    'grid.linewidth': 0.5,
    'axes.grid': True,                # 启用网格
    'axes.grid.axis': 'y',            # 仅显示横向网格线
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'axes.edgecolor': "black",        # 边框颜色
    'axes.linewidth': 1.2,
})
mpl.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
mpl.rcParams['font.size'] = 18  # 设置基本字体大小为 12 点
mpl.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签的字体大小
mpl.rcParams['axes.titlesize'] = 20  # 设置坐标轴标题的字体大小
mpl.rcParams['xtick.labelsize'] = 16  # 设置x轴刻度标签的字体大小
mpl.rcParams['ytick.labelsize'] = 16  # 设置y轴刻度标签的字体大小
mpl.rcParams['legend.fontsize'] = 15  # 设置图例的字体大小

# ALPHA = 0.2


data = {
    "Group": [
        "RL+LLM", "RL+LLM", "RL+Human", "RL+Human",
        "LLM+Human", "LLM+Human", "RL+RL", "LLM+LLM", "Human+Human"
    ],
    "Mean": [0.85, 0.82, 0.75, 0.71, 0.78, 0.92, 0.90, 0.86, 0.88],
    "Std": [0.04, 0.05, 0.06, 0.07, 0.06, 0.03, 0.04, 0.05, 0.03]
}

df = pd.DataFrame(data)

# 分组信息
groups = ["RL+LLM", "RL+Human", "LLM+Human", "RL+RL", "LLM+LLM", "Human+Human"]
bar_width = 0.4
index = np.arange(len(groups))

# 颜色设置
colors = ['#5E8DAA', '#E29742']

plt.figure(figsize=(12, 6))

# 存储已添加的标签，避免重复显示图例
legend_added = set()

for i, group in enumerate(groups):
    sub_df = df[df["Group"] == group]
    
    if len(sub_df) == 1:
        # 自合作情况，只画一个柱子
        plt.bar(index[i], sub_df.iloc[0]['Mean'], yerr=sub_df.iloc[0]['Std'],
                width=bar_width, color='gray', edgecolor='black',
                capsize=5, label="Self-Cooperation" if "Self-Cooperation" not in legend_added else "")
        legend_added.add("Self-Cooperation")
    else:
        # 两个方向的合作
        plt.bar(index[i] - bar_width/2, sub_df.iloc[0]['Mean'], yerr=sub_df.iloc[0]['Std'],
                width=bar_width, color=colors[0], edgecolor='black', capsize=5,
                # label=sub_df.iloc[0]['Order'] if "Main" not in legend_added else ""
                )
        plt.bar(index[i] + bar_width/2, sub_df.iloc[1]['Mean'], yerr=sub_df.iloc[1]['Std'],
                width=bar_width, color=colors[1], edgecolor='black', hatch='//', alpha=0.7,
                capsize=5, 
                label="Reverse position" if "Reverse" not in legend_added else ""
                )
        legend_added.add("Main")
        legend_added.add("Reverse")

# 美化图表
plt.xticks(index, groups)
plt.ylabel("Cooperation Performance", fontsize=14)
# plt.title("Cross-Agent and Self-Agent Cooperation Performance", fontsize=16)
plt.ylim(0.6, 1.0)
plt.tight_layout()

# 添加图例
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# del by_label['RL→LLM']

plt.legend(by_label.values(), by_label.keys(), loc='upper right')

# 保存图像
plt.savefig("cross_agent_collaboration_supereasy.pdf", dpi=300, bbox_inches='tight')
plt.show()