import torch
import torch.nn as nn
from mamba_layer import MambaBlock

class LumbarCNNMamba(nn.Module):
    def __init__(self, input_channels=4, d_model=128, dropout_rate=0.1): # 修改点：dropout_rate 降至 0.1
        super().__init__()

        self.cnn_backbone = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )

        self.mamba1 = MambaBlock(d_model=d_model, d_state=16, expand=2)
        self.mamba2 = MambaBlock(d_model=d_model, d_state=16, expand=2)
        self.layer_norm = nn.LayerNorm(d_model)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)  # 输出 [腰部角度, 腰部扭矩]
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x_cnn = self.cnn_backbone(x)
        x_cnn = x_cnn.transpose(1, 2)

        x_seq = self.mamba1(x_cnn)
        x_seq = self.mamba2(x_seq)
        x_seq = self.layer_norm(x_seq + x_cnn)

        last_step_feature = x_seq[:, -1, :]
        return self.regressor(last_step_feature)