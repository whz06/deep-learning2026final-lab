from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.head(self.dropout(y[:, -1, :])).squeeze(-1)


class CNNRegressor(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, filters: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=2)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=2)
        self.conv3 = nn.Conv1d(filters, filters, kernel_size=2)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        out_len = seq_len - 3
        if out_len <= 0:
            raise ValueError("CNN 的 seq_len 至少需要 4")
        self.head = nn.Linear(filters * out_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.dropout(self.act(self.conv1(x)))
        x = self.dropout(self.act(self.conv2(x)))
        x = self.dropout(self.act(self.conv3(x)))
        return self.head(torch.flatten(x, start_dim=1)).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, filters: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=filters, hidden_size=hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = x.transpose(1, 2)
        y, _ = self.lstm(x)
        return self.head(self.dropout(y[:, -1, :])).squeeze(-1)


def build_model(model_name: str, input_dim: int, seq_len: int, hidden: int, filters: int, dropout: float) -> nn.Module:
    if model_name == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden=hidden, dropout=dropout)
    if model_name == "cnn":
        return CNNRegressor(input_dim=input_dim, seq_len=seq_len, filters=filters, dropout=dropout)
    if model_name == "cnn_lstm":
        return CNNLSTMRegressor(input_dim=input_dim, filters=filters, hidden=hidden, dropout=dropout)
    raise ValueError(f"未知模型: {model_name}")


def pick_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

