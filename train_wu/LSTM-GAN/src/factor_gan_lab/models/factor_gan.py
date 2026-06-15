from __future__ import annotations

"""
Factor-GAN 的模型结构定义。

这里保留的是"模型本身", 而不是任何特定因子集合。
也就是说:
- 你可以继续使用 LSTM 生成器 + LSTM 判别器这套结构
- 但输入因子维度 `n_factors` 不再固定为旧版 19
"""

from dataclasses import dataclass

import torch
from torch import nn


class GeneratorLSTM(nn.Module):
    """
    生成器 G: 纯 LSTM。

    T 步因子序列 → LSTM → h_last → Linear → r_hat

    输入:
    - `z`: `(B, T, F)` — 因子序列

    输出:
    - `r_hat`: `(B,)` — 每个样本一个预测收益
    """

    def __init__(
        self,
        n_factors: int = 26,
        hidden_size: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_factors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers <= 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :]).squeeze(-1)


class DiscriminatorLSTM(nn.Module):
    """
    判别器 D: 方式 B

    LSTM 只看纯因子序列, 不提前接收收益信息。
    LSTM 编码完整 T 天因子走势后, 将隐藏状态与预测收益拼接,
    再由 FC 打分: "给定这段时间的因子走势, 这个收益合理吗?"。

    输入:
    - `z`: `(B, T, F)` — 纯因子序列
    - `returns`: `(B,)` — 预测/真实收益

    输出:
    - `score`: `(B,)`, WGAN 风格实数分数 (不接 sigmoid)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # head 输入: h_last (hidden) + returns (1) = hidden + 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + 1, hidden_size * 2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, z: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        # z: (B, T, F), returns: (B,)
        out, _ = self.lstm(z)
        h_last = out[:, -1, :]  # (B, hidden)
        combined = torch.cat([h_last, returns.unsqueeze(-1)], dim=-1)  # (B, hidden+1)
        return self.head(combined).squeeze(-1)  # (B,)


@dataclass(frozen=True)
class FactorGANConfig:
    """
    Factor-GAN 的结构超参数。

    G: LSTM(256,2层) → Linear(64) → r_hat
    D: LSTM(128,2层) → [h_last, returns] → score
    """

    n_factors: int
    g_hidden: int = 256
    g_layers: int = 2
    timestep: int = 5
    d_hidden: int = 128
    d_layers: int = 2
    dropout: float = 0.1
    negative_slope: float = 0.2
    gp_lambda: float = 10.0


class FactorGAN(nn.Module):
    """
    把生成器 G 与判别器 D 组合到一起。

    你在训练时通常会分开调用:
    - `r_hat = G(z)`
    - `D([factors, r_real])`
    - `D([factors, r_hat])`

    但在 smoke test 里, 组合成一个类会更方便。
    """

    def __init__(self, cfg: FactorGANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.G = GeneratorLSTM(
            n_factors=cfg.n_factors,
            hidden_size=cfg.g_hidden,
            num_layers=cfg.g_layers,
        )
        self.D = DiscriminatorLSTM(
            input_size=cfg.n_factors,
            hidden_size=cfg.d_hidden,
            num_layers=cfg.d_layers,
            dropout=cfg.dropout,
            negative_slope=cfg.negative_slope,
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 3:
            raise ValueError(f"`z` 必须是三维张量 `(B, T, F)`, 当前收到 {tuple(z.shape)}")
        r_hat = self.G(z)
        d_score = self.D(z, r_hat)  # 方式 B: 分别传入因子序列和收益
        return r_hat, d_score
