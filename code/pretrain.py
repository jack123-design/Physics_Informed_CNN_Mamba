import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VRDLRDataset
from model import LumbarCNNMamba
import joblib  # 引入 joblib


def main():
    BASE_DIR = "/mnt/e/深度学习项目/基于物理信息 CNN-Mamba 的躯干运动意图识别与动态力矩估计/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预训练受试者列表 (已匿名化)
    train_subjects = ['subject1', 'subject2', 'subject3']

    print("加载通用训练集...")
    train_dataset = VRDLRDataset(BASE_DIR, subjects=train_subjects, is_train=True)
    if len(train_dataset) == 0:
        print("错误：数据集样本为0，请检查路径。")
        return

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    model = LumbarCNNMamba().to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            preds = model(x)
            loss = criterion(preds, y)

            if torch.isnan(loss):
                print(f"[警告] Epoch {epoch + 1} 检测到 Loss 为 NaN，跳过该批次")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Pretrain Epoch [{epoch + 1}/{epochs}] | General MSE Loss: {avg_train:.4f}")

    torch.save(model.state_dict(), "general_model.pth")
    print("通用模型预训练完成，权重已保存。")

    # 【必须添加】：保存 Scaler 给迁移训练使用
    joblib.dump(train_dataset.scaler, "general_scaler.pkl")
    print("通用 Scaler 已保存。")


if __name__ == "__main__":
    main()