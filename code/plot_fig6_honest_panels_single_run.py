import numpy as np
import matplotlib.pyplot as plt
import os

# 全局极简学术规范
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10

SAVE_DIR = r'./figures_final'
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Fig 6: 双面板敏感性分析 (纯单次实验展示，无视觉截断，无虚假误差棒)
# =============================================================================
def plot_fig6_honest_panels_single_run():
    lambdas = ['0.0', '0.01', '0.05', '0.1', '0.25', '0.5', '1.0']
    x_pos = np.arange(len(lambdas))

    # 填入经严格审计的单次运行数据 (Single run)
    angle_r2_single_run = np.array([0.7791, 0.7791, 0.7790, 0.7789, 0.7786, 0.7778, 0.7767])
    physics_mse_single_run = np.array([19.3, 19.1, 18.5, 17.8, 16.1, 14.0, 11.2])

    # 采用上下对齐的双面板 (Vertically aligned panels)，共享横轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, dpi=300)

    # ---------------------------------------------------------
    # (a) Panel 1: Angle R2
    # ---------------------------------------------------------
    ax1.plot(x_pos, angle_r2_single_run, marker='o', markersize=6, color='#333333', linewidth=1.5)
    ax1.set_ylabel(r'Angle Accuracy ($R^2$)')
    ax1.set_title('(a) Kinematic Estimation Accuracy', loc='left')

    # 严格的 0-1.0 物理范围，避免截断导致视觉夸大
    ax1.set_ylim([0, 1.0])

    # 在图中右上角标注 R2 的微弱变化
    ax1.text(0.95, 0.65, r"$\Delta R^2_{max} = -0.0024$ ($\approx -0.31\%$)",
             transform=ax1.transAxes, ha='right', fontsize=10, color='#333333')

    # ---------------------------------------------------------
    # (b) Panel 2: Physics MSE
    # ---------------------------------------------------------
    ax2.plot(x_pos, physics_mse_single_run, marker='s', markersize=6, color='#333333', linewidth=1.5)
    ax2.set_xlabel(r'Physics-Loss Weight ($\lambda_p$)')
    ax2.set_ylabel(r'Consistency Error ($MSE_{cons}$)')
    ax2.set_title('(b) Output-Space Physical Consistency Error', loc='left')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(lambdas)
    ax2.set_ylim([0, 22])  # 留出空间

    # 标注一致性误差的大幅下降 (42.0%)
    # 起点在 lambda=1.0 (x=6)，终点指向 lambda=0.0 (x=0) 方向
    # 删掉宽泛的 width, headwidth, shrink 参数，使用自带的 '->' 样式
    ax2.annotate(r'$-42.0\%$', xy=(6, 11.2), xytext=(4.5, 14),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5),
                 fontsize=10, fontweight='bold', color='#333333')

    # ---------------------------------------------------------
    # 全局排版美化
    # ---------------------------------------------------------
    for ax in [ax1, ax2]:
        # 极淡的网格辅助线
        ax.grid(True, linestyle=':', alpha=0.4)
        # 去掉顶部和右侧的边框 (Spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 导出矢量图(PDF)和高分辨率位图(PNG)
    plt.savefig(os.path.join(SAVE_DIR, 'Fig6_PhysicsSensitivity.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(SAVE_DIR, 'Fig6_PhysicsSensitivity.png'), format='png', dpi=600, bbox_inches='tight')
    plt.close()

    print("[SUCCESS] Fig 6 (双面板物理一致性图 - 严谨版) 已生成。")


if __name__ == '__main__':
    plot_fig6_honest_panels_single_run()