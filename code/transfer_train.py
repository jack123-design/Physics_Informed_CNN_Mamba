import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VRDLRDataset
from model import LumbarCNNMamba
import numpy as np
import joblib


class PhysicsInformedTransferLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        pred_theta, pred_tau = preds[:, 0], preds[:, 1]
        target_theta, target_tau = targets[:, 0], targets[:, 1]

        mse_theta = self.mse(pred_theta, target_theta)
        mse_tau = self.mse(pred_tau, target_tau)

        # 截断物理梯度回流，用预测的角度去计算理论扭矩
        real_theta = pred_theta.detach()
        phys_tau_real = self.calculate_physics_torque(real_theta)

        # 直接在原始量级计算物理一致性损失
        loss_phy = self.mse(phys_tau_real, pred_tau)

        total_loss = mse_theta + mse_tau + self.alpha * loss_phy
        return total_loss, loss_phy

    def calculate_physics_torque(self, theta):
        m_torso = 45.0
        g = 9.81
        l_com = 0.30

        theta_rad = theta * (np.pi / 180.0)
        tau_theoretical = m_torso * g * l_com * torch.sin(theta_rad)
        return tau_theoretical


def main():
    BASE_DIR = "/mnt/e/深度学习项目/基于物理信息 CNN-Mamba 的躯干运动意图识别与动态力矩估计/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 迁移学习目标受试者 (已匿名化)
    target_subject = ['subject8']

    print(f"加载目标受试者 {target_subject[0]} 的迁移数据集...")

    # 加载预训练的 scaler，防止特征空间偏移，并将 is_train 设为 False
    pretrained_scaler = joblib.load("general_scaler.pkl")
    transfer_dataset = VRDLRDataset(
        BASE_DIR,
        subjects=target_subject,
        is_train=False,
        scaler=pretrained_scaler
    )

    if len(transfer_dataset) == 0:
        print("错误：迁移数据集样本为0。")
        return

    transfer_loader = DataLoader(transfer_dataset, batch_size=32, shuffle=True, num_workers=0)

    model = LumbarCNNMamba().to(device).float()
    model.load_state_dict(torch.load("general_model.pth"))

    # 打破均值坍塌，重置回归头
    for m in model.regressor.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    criterion = PhysicsInformedTransferLoss(alpha=0.25)

    # 分层学习率：骨干网络冻结式微调 (1e-5)，回归头全速学习 (1e-3)
    optimizer = torch.optim.AdamW([
        {'params': model.cnn_backbone.parameters(), 'lr': 1e-5},
        {'params': model.mamba1.parameters(), 'lr': 1e-5},
        {'params': model.mamba2.parameters(), 'lr': 1e-5},
        {'params': model.layer_norm.parameters(), 'lr': 1e-5},
        {'params': model.regressor.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    epochs = 60

    for epoch in range(epochs):
        model.train()
        epoch_total_loss = 0
        epoch_phy_loss = 0

        for x, y in transfer_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            preds = model(x)

            loss, loss_phy = criterion(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_phy_loss += loss_phy.item()

        scheduler.step()

        avg_loss = epoch_total_loss / len(transfer_loader)
        avg_phy = epoch_phy_loss / len(transfer_loader)

        if (epoch + 1) % 5 == 0:
            print(f"Transfer Epoch [{epoch + 1}/{epochs}] | Total Loss: {avg_loss:.4f} | Physics Loss: {avg_phy:.4f}")

    torch.save(model.state_dict(), f"personalized_model_{target_subject[0]}.pth")
    print("个性化物理约束迁移训练完成！")


if __name__ == "__main__":
    main()