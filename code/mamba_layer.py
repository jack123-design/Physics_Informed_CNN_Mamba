import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=False
        )

        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # --- 标准投影层初始化 ---
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # 【修复核心 1】：严格遵循 Mamba 论文的 dt 步长初始化，防止梯度爆炸
        dt_init_std = d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_init = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt_init = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt_init)

        # 【修复核心 2】：A 矩阵必须用特定的对数域递增初始化，保证记忆的长期稳定
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log.data.copy_(torch.log(A))

        nn.init.constant_(self.D, 1.0)

        self.d_state = d_state
        self.d_inner = d_inner

    def forward(self, x):
        residual = x
        x = self.norm(x)

        x, z = self.in_proj(x).chunk(2, dim=-1)

        seq_len = x.size(1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len]
        x = F.silu(x.transpose(1, 2))

        delta, B, C = torch.split(
            self.x_proj(x), [1, self.d_state, self.d_state], dim=-1
        )

        # 【修复核心 3】：绝对禁止在这里调用 F.softplus！
        # 后续的 selective_scan_fn 开启了 delta_softplus=True，会在 CUDA 底层自动计算。
        dt = self.dt_proj(delta)

        A = -torch.exp(self.A_log)

        y = selective_scan_fn(
            u=x.transpose(1, 2),
            delta=dt.transpose(1, 2),
            A=A,
            B=B.transpose(1, 2),
            C=C.transpose(1, 2),
            D=self.D,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False
        ).transpose(1, 2)

        return self.out_proj(y * F.silu(z)) + residual