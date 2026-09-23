import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 引入自定义模块 (需确保 dataset.py 和 model.py 的实现与此解耦)
from dataset import VRDLRDataset
from model import LumbarCNNMamba

# =============================================================================
# [全局设置] 物理参数统一管理与 BSPC 期刊绘图规范
# =============================================================================
PHYSICS_PARAMS = {
    "m": 45.0,  # Equivalent effective mass (kg)
    "g": 9.81,  # Gravitational acceleration (m/s^2)
    "l": 0.30   # Effective moment arm (m)
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 11


# =============================================================================
# 1. 评估函数：载入模型并输出全量预测
# =============================================================================
def evaluate_model(model_path, data_loader, device):
    print(f"\n[评估] 正在载入模型: {os.path.basename(model_path)}")
    model = LumbarCNNMamba().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    return all_targets, all_preds


# =============================================================================
# 2. 生成 Fig 4: 连续时序追踪图 (极简学术风)
# =============================================================================
def plot_fig4_temporal_tracking(true_y, pred_y, start_idx=0, duration_s=15.0, fps=100, subject_id="Unknown",
                                trial_id="Unknown", save_dir='.'):
    num_samples = int(duration_s * fps)
    end_idx = start_idx + num_samples

    if end_idx > len(true_y):
        end_idx = len(true_y)
        num_samples = end_idx - start_idx

    time = np.arange(num_samples) / float(fps)

    ref_angle, est_angle = true_y[start_idx:end_idx, 0], pred_y[start_idx:end_idx, 0]
    ref_torque, est_torque = true_y[start_idx:end_idx, 1], pred_y[start_idx:end_idx, 1]

    metrics = {
        'ang_r2': r2_score(ref_angle, est_angle),
        'ang_rmse': np.sqrt(mean_squared_error(ref_angle, est_angle)),
        'ang_mae': mean_absolute_error(ref_angle, est_angle),
        'tau_r2': r2_score(ref_torque, est_torque),
        'tau_rmse': np.sqrt(mean_squared_error(ref_torque, est_torque)),
        'tau_mae': mean_absolute_error(ref_torque, est_torque)
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # (a) 运动学角度追踪
    ax1.plot(time, ref_angle, color='#333333', linewidth=1.5, linestyle='-', label='Reference')
    # 降低预测线宽度并增加透明度，避免遮挡真实的 Reference 趋势
    ax1.plot(time, est_angle, color='#1f77b4', linewidth=1.2, linestyle='--', alpha=0.85, label='Estimated')
    ax1.set_ylabel(r'Lumbar Angle ($^\circ$)')
    ax1.set_title('(a) Temporal Tracking of Lumbar Kinematics', loc='left', fontweight='semibold')
    # 将图例移至右下角防止遮挡曲线波峰
    ax1.legend(loc='lower right', frameon=False, fontsize=10)

    text_ang = f"$R^2$: {metrics['ang_r2']:.2f}\nRMSE: {metrics['ang_rmse']:.2f}$^\circ$\nMAE: {metrics['ang_mae']:.2f}$^\circ$"
    ax1.text(0.02, 0.92, text_ang, transform=ax1.transAxes, fontsize=10, verticalalignment='top')

    # (b) 等效重力矩追踪
    ax2.plot(time, ref_torque, color='#333333', linewidth=1.5, linestyle='-', label='Reference')
    ax2.plot(time, est_torque, color='#1f77b4', linewidth=1.2, linestyle='--', alpha=0.85, label='Estimated')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Equivalent Torque (N$\cdot$m)')
    ax2.set_title('(b) Temporal Tracking of Equivalent Gravitational Torque', loc='left', fontweight='semibold')
    ax2.legend(loc='lower right', frameon=False, fontsize=10)

    text_tau = f"$R^2$: {metrics['tau_r2']:.2f}\nRMSE: {metrics['tau_rmse']:.2f} N$\cdot$m\nMAE: {metrics['tau_mae']:.2f} N$\cdot$m"
    ax2.text(0.02, 0.92, text_tau, transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    base_filename = os.path.join(save_dir, 'Fig4_TemporalTracking')

    plt.savefig(f'{base_filename}.pdf', format='pdf', bbox_inches='tight',
                metadata={'Creator': 'Python/Matplotlib', 'Title': 'Temporal Tracking of Lumbar Kinematics'})
    plt.savefig(f'{base_filename}.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

    pd.DataFrame({
        'Subject': [subject_id] * num_samples,
        'Trial_ID': [trial_id] * num_samples,
        'Time_s': time,
        'Ref_Angle_deg': ref_angle, 'Est_Angle_deg': est_angle,
        'Ref_Torque_Nm': ref_torque, 'Est_Torque_Nm': est_torque
    }).to_csv(f'{base_filename}_data.csv', index=False)
    print(f"[SUCCESS] Fig 4 生成完毕 (PDF/PNG/CSV)。")


# =============================================================================
# 3. 生成 Fig 5: 物理一致性消融对比图 (Baseline vs Physics-informed)
# =============================================================================
def plot_fig5_physics_consistency(pred_y_base, pred_y_phys, save_dir='.', max_points=2000):
    m, g, l = PHYSICS_PARAMS['m'], PHYSICS_PARAMS['g'], PHYSICS_PARAMS['l']

    # 1. 提取全量推导力矩与估计力矩
    base_est_angle, base_est_torque = pred_y_base[:, 0], pred_y_base[:, 1]
    phys_est_angle, phys_est_torque = pred_y_phys[:, 0], pred_y_phys[:, 1]

    base_derived_torque = m * g * l * np.sin(np.deg2rad(base_est_angle))
    phys_derived_torque = m * g * l * np.sin(np.deg2rad(phys_est_angle))

    # 2. 计算与正文严谨对齐的 MSE_cons
    metrics_base = {
        'R2_cons': r2_score(base_derived_torque, base_est_torque),
        'MSE_cons': mean_squared_error(base_derived_torque, base_est_torque)
    }
    metrics_phys = {
        'R2_cons': r2_score(phys_derived_torque, phys_est_torque),
        'MSE_cons': mean_squared_error(phys_derived_torque, phys_est_torque)
    }

    # 3. 确定性降采样
    total_len = len(base_est_angle)
    if total_len > max_points:
        idx = np.linspace(0, total_len - 1, max_points, dtype=int)
    else:
        idx = np.arange(total_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

    # 收紧坐标轴范围，消除大面积留白 (-25 调整为实际数据的下界)
    min_val = -5.0
    max_val = max(np.max(phys_derived_torque), np.max(phys_est_torque),
                  np.max(base_derived_torque), np.max(base_est_torque)) + 10.0

    # (a) Baseline Panel
    ax1.scatter(base_derived_torque[idx], base_est_torque[idx], alpha=0.25, color='#2ca02c', edgecolor='none', s=8)
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal consistency ($y=x$)')
    ax1.set_xlabel(r'Angle-derived torque $mgl\sin(\hat{\theta})$ (N$\cdot$m)')
    ax1.set_ylabel(r'Estimated torque $\hat{\tau}_{eq}$ (N$\cdot$m)')
    ax1.set_title('(a) Baseline CNN-Mamba', loc='left', fontweight='semibold')
    ax1.set_xlim([min_val, max_val])
    ax1.set_ylim([min_val, max_val])

    # 统一变更为 MSE_cons，并带入正确的平方单位
    txt_base = f"$R^2_{{cons}}$: {metrics_base['R2_cons']:.4f}\n$MSE_{{cons}}$: {metrics_base['MSE_cons']:.1f} (N$\cdot$m)$^2$"
    ax1.text(0.05, 0.95, txt_base, transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    ax1.legend(loc='lower right', frameon=False)

    # (b) Physics-informed Panel
    ax2.scatter(phys_derived_torque[idx], phys_est_torque[idx], alpha=0.25, color='#2ca02c', edgecolor='none', s=8)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal consistency ($y=x$)')
    ax2.set_xlabel(r'Angle-derived torque $mgl\sin(\hat{\theta})$ (N$\cdot$m)')
    ax2.set_title('(b) Physics-informed CNN-Mamba', loc='left', fontweight='semibold')

    txt_phys = f"$R^2_{{cons}}$: {metrics_phys['R2_cons']:.4f}\n$MSE_{{cons}}$: {metrics_phys['MSE_cons']:.1f} (N$\cdot$m)$^2$"
    ax2.text(0.05, 0.95, txt_phys, transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    base_filename = os.path.join(save_dir, 'Fig5_PhysicsConsistency')
    plt.savefig(f'{base_filename}.pdf', format='pdf', bbox_inches='tight',
                metadata={'Creator': 'Python/Matplotlib', 'Title': 'Output-Space Biomechanical Consistency'})
    plt.savefig(f'{base_filename}.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

    pd.DataFrame({
        'Base_Angle_Derived_Torque_Nm': base_derived_torque, 'Base_Estimated_Torque_Nm': base_est_torque,
        'Phys_Angle_Derived_Torque_Nm': phys_derived_torque, 'Phys_Estimated_Torque_Nm': phys_est_torque
    }).to_csv(f'{base_filename}_full_data.csv', index=False)
    print(f"[SUCCESS] Fig 5 生成完毕 (PDF/PNG/CSV)。")


def main():
    BASE_DIR = "/mnt/e/深度学习项目/基于物理信息 CNN-Mamba 的躯干运动意图识别与动态力矩估计/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 目标受试者 (已匿名化)
    target_subject = ['subject8']

    TRIAL_START_INDEX = 0
    TRIAL_ID = "Trial_03"

    print("=" * 60)
    print("🚀 [1/3] 载入测试集以进行评估 (强制 shuffle=False 保护时序连续性)...")
    try:
        pretrained_scaler = joblib.load("general_scaler.pkl")
    except FileNotFoundError:
        print("[错误] 未找到 general_scaler.pkl，请检查预训练阶段输出！")
        return

    test_dataset = VRDLRDataset(BASE_DIR, subjects=target_subject, is_train=False, scaler=pretrained_scaler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print("=" * 60)
    print("🚀 [2/3] 提取模型推理输出...")

    base_model_path = "general_model.pth"
    phys_model_path = f"personalized_model_{target_subject[0]}.pth"

    if not os.path.exists(phys_model_path):
        print(f"[错误] 缺失微调模型 {phys_model_path}！")
        return

    _, pred_y_base = evaluate_model(base_model_path, test_loader, device)
    true_y, pred_y_phys = evaluate_model(phys_model_path, test_loader, device)

    print("=" * 60)
    print("🚀 [3/3] 正在生成符合 BSPC 标准的矢量图像与 CSV...")

    plot_fig4_temporal_tracking(
        true_y, pred_y_phys,
        start_idx=TRIAL_START_INDEX,
        duration_s=15.0,
        fps=100,
        subject_id=target_subject[0],
        trial_id=TRIAL_ID
    )

    plot_fig5_physics_consistency(pred_y_base, pred_y_phys)

if __name__ == "__main__":
    main()