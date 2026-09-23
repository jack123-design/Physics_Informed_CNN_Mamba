import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import VRDLRDataset
from model import LumbarCNNMamba
import joblib


def main():
    BASE_DIR = "/mnt/e/深度学习项目/基于物理信息 CNN-Mamba 的躯干运动意图识别与动态力矩估计/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 评估受试者 (已匿名化)
    test_subject = ['subject8']

    print(f"正在加载受试者 {test_subject[0]} 的评估数据进行绘图...")

    pretrained_scaler = joblib.load("general_scaler.pkl")
    test_dataset = VRDLRDataset(
        BASE_DIR,
        subjects=test_subject,
        is_train=False,
        scaler=pretrained_scaler
    )

    # shuffle=False 极其重要，必须保证时序连贯性才能画出正确的曲线
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = LumbarCNNMamba().to(device).float()
    model.load_state_dict(torch.load(f"personalized_model_{test_subject[0]}.pth"))
    model.eval()

    all_preds = []
    all_targets = []

    print("执行前向推理提取数据...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            x = torch.clamp(x, min=-5.0, max=5.0)

            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    pred_angle, pred_torque = all_preds[:, 0], all_preds[:, 1]
    target_angle, target_torque = all_targets[:, 0], all_targets[:, 1]

    print("数据提取完毕，开始渲染论文级高清图像 (300 DPI)...")

    # ==========================================
    # 图 1：时序跟随曲线对比图 (Time-Series Tracking)
    # ==========================================
    # 为了图像清晰不拥挤，我们截取其中连续的 30 秒数据展示 (30s * 100Hz = 3000 个采样点)
    display_points = min(3000, len(pred_angle))
    time_axis = np.arange(display_points) * 0.01  # 步长 0.01 秒 (基于 100Hz 的特征频率)

    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制腰部角度曲线
    axes1[0].plot(time_axis, target_angle[:display_points], label='Ground Truth', color='#1f77b4', linewidth=2)
    axes1[0].plot(time_axis, pred_angle[:display_points], label='Estimation', color='#ff7f0e', linestyle='--',
                  linewidth=2)
    axes1[0].set_ylabel('Lumbar Angle (°)', fontsize=14, fontweight='bold')
    axes1[0].legend(loc='upper right', fontsize=12)
    axes1[0].grid(True, linestyle=':', alpha=0.7)
    axes1[0].tick_params(axis='both', labelsize=12)

    # 绘制腰部力矩曲线
    axes1[1].plot(time_axis, target_torque[:display_points], label='Ground Truth', color='#1f77b4', linewidth=2)
    axes1[1].plot(time_axis, pred_torque[:display_points], label='Estimation', color='#ff7f0e', linestyle='--',
                  linewidth=2)
    axes1[1].set_ylabel('Lumbar Torque (Nm)', fontsize=14, fontweight='bold')
    axes1[1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    axes1[1].legend(loc='upper right', fontsize=12)
    axes1[1].grid(True, linestyle=':', alpha=0.7)
    axes1[1].tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    fig1.savefig('tracking_curve.png', dpi=300, bbox_inches='tight')
    print(" -> 已保存时序曲线图: tracking_curve.png")

    # ==========================================
    # 图 2：相关性拟合散点图 (Correlation Scatter)
    # ==========================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # 角度散点
    axes2[0].scatter(target_angle, pred_angle, alpha=0.1, color='#1f77b4', s=10)
    min_a, max_a = min(target_angle.min(), pred_angle.min()), max(target_angle.max(), pred_angle.max())
    axes2[0].plot([min_a, max_a], [min_a, max_a], 'r--', linewidth=2, label='Ideal Fit')
    axes2[0].set_title('Correlation of Lumbar Angle', fontsize=14, fontweight='bold')
    axes2[0].set_xlabel('Ground Truth Angle (°)', fontsize=12)
    axes2[0].set_ylabel('Estimated Angle (°)', fontsize=12)
    axes2[0].grid(True, linestyle=':', alpha=0.7)
    axes2[0].legend(loc='upper left', fontsize=12)

    # 力矩散点
    axes2[1].scatter(target_torque, pred_torque, alpha=0.1, color='#2ca02c', s=10)
    min_t, max_t = min(target_torque.min(), pred_torque.min()), max(target_torque.max(), pred_torque.max())
    axes2[1].plot([min_t, max_t], [min_t, max_t], 'r--', linewidth=2, label='Ideal Fit')
    axes2[1].set_title('Correlation of Lumbar Torque', fontsize=14, fontweight='bold')
    axes2[1].set_xlabel('Ground Truth Torque (Nm)', fontsize=12)
    axes2[1].set_ylabel('Estimated Torque (Nm)', fontsize=12)
    axes2[1].grid(True, linestyle=':', alpha=0.7)
    axes2[1].legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    fig2.savefig('correlation_scatter.png', dpi=300, bbox_inches='tight')
    print(" -> 已保存散点相关图: correlation_scatter.png")


if __name__ == "__main__":
    main()