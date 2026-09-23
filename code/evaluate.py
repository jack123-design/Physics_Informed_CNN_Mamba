import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import VRDLRDataset
from model import LumbarCNNMamba
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    BASE_DIR = "/mnt/e/深度学习项目/基于物理信息 CNN-Mamba 的躯干运动意图识别与动态力矩估计/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 评估目标受试者 (已匿名化)
    test_subject = ['subject8']

    print(f"加载目标受试者 {test_subject[0]} 的评估数据集...")

    # 加载预训练的 Scaler (必须保证特征工程环境与训练时完全一致)
    pretrained_scaler = joblib.load("general_scaler.pkl")

    test_dataset = VRDLRDataset(
        BASE_DIR,
        subjects=test_subject,
        is_train=False,
        scaler=pretrained_scaler
    )

    if len(test_dataset) == 0:
        print("错误：评估数据集样本为0。")
        return

    # 测试阶段 batch_size 可以开大，且不需要 shuffle，保证时序顺序
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 实例化并加载刚刚训练好的个性化模型
    model = LumbarCNNMamba().to(device).float()
    model.load_state_dict(torch.load(f"personalized_model_{test_subject[0]}.pth"))

    # 【核心】：一定要切换到 eval 模式，关闭 Dropout 和 BatchNorm 的动态更新
    model.eval()

    all_preds = []
    all_targets = []

    print("开始进行全量前向推理...")
    # 禁用梯度计算，加速推理并节省显存
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # 防爆截断与训练时保持绝对一致
            x = torch.clamp(x, min=-5.0, max=5.0)

            preds = model(x)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 拼接所有批次的数据
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 分离角度和扭矩
    pred_angle, pred_torque = all_preds[:, 0], all_preds[:, 1]
    target_angle, target_torque = all_targets[:, 0], all_targets[:, 1]

    # 计算定量评估指标
    print("\n" + "=" * 45)
    print(f" 🚀 个性化模型 ({test_subject[0]}) 最终评估指标")
    print("=" * 45)

    # 1. 角度评估
    rmse_angle = np.sqrt(mean_squared_error(target_angle, pred_angle))
    mae_angle = mean_absolute_error(target_angle, pred_angle)
    r2_angle = r2_score(target_angle, pred_angle)

    print(f"[腰部角度 θ]")
    print(f"  - RMSE (均方根误差): {rmse_angle:.4f} °")
    print(f"  - MAE  (平均绝对误差): {mae_angle:.4f} °")
    print(f"  - R²   (决定系数):     {r2_angle:.4f}")
    print("-" * 45)

    # 2. 扭矩评估
    rmse_torque = np.sqrt(mean_squared_error(target_torque, pred_torque))
    mae_torque = mean_absolute_error(target_torque, pred_torque)
    r2_torque = r2_score(target_torque, pred_torque)

    print(f"[动态力矩 τ]")
    print(f"  - RMSE (均方根误差): {rmse_torque:.4f} Nm")
    print(f"  - MAE  (平均绝对误差): {mae_torque:.4f} Nm")
    print(f"  - R²   (决定系数):     {r2_torque:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    main()