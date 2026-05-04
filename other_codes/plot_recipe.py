import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据
df = pd.read_csv('/users/PAS2099/mino/ICICLE/csv/camera-trap-CVPR - ECCV (FINAL) (1).csv')

# 提取列并清洗空值
x_col = 'Accum (new)'
y_col = 'Best Accum (new)'
plot_data = df[[x_col, y_col]].dropna()

x = plot_data[x_col]
y = plot_data[y_col]

# 2. 设置画布
fig, ax = plt.subplots(figsize=(8, 7))

# 点的颜色：浅蓝灰色 (#ADD8E6)，带黑色细边框
ax.scatter(x, y, color='blue', edgecolors='black', linewidths=0.5, 
           s=60, alpha=0.9, label='Cameras', zorder=3)

# 红色 y=x 线：实线，加粗
line_coords = np.linspace(0, 1, 100)
ax.plot(line_coords, line_coords, color='#FF0000', linestyle='-', 
        linewidth=2, label='y=x', zorder=2)

# 3. 坐标轴线加粗 (Spines)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 4. 标题和标签加粗
ax.set_xlim(0, 1.0)
ax.set_ylim(0.6, 1.0)

# 显式指定 X 轴和 Y 轴的刻度点
x_ticks = np.arange(0, 1.1, 0.2)   # 0.0, 0.2, ..., 1.0
y_ticks = np.arange(0.6, 1.01, 0.1) # 0.6, 0.7, 0.8, 0.9, 1.0

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# 设置刻度线粗细
ax.tick_params(axis='both', which='major', width=2.5, labelsize=11)

# 6. 【关键点】将刻度数字设置为粗体
ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks], fontweight='bold')
ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontweight='bold')

# 6. 网格线：浅灰色虚线
ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

# 7. 图例加粗
ax.legend(loc='lower right', frameon=True, prop={'weight': 'bold'})

plt.tight_layout()
plt.savefig('/users/PAS2099/mino/ICICLE/plots2/recipe_accum.png', dpi=300)
print(f"Plot saved to /users/PAS2099/mino/ICICLE/plots2/recipe_accum.png.")
plt.show()