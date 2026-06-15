from __future__ import annotations

"""
Factor-GAN 的训练引擎。

支持：
- 日频滚动窗口切分（train_days / val_days / test_days）
- WGAN-GP 梯度惩罚
- T 日序列加载（LSTM 时序输入）
- 验证集早停
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from factor_gan_lab.data import build_sequences
from factor_gan_lab.models import FactorGAN, FactorGANConfig


def pick_device(device_arg: str) -> torch.device:
    """选择训练设备。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_loader(
    df: pd.DataFrame,
    factor_columns: list[str],
    batch_size: int,
    shuffle: bool,
    timestep: int = 5,
    windows_dir: str | None = None,
    date_range: list[pd.Timestamp] | None = None,
) -> DataLoader:
    if windows_dir is not None:
        from factor_gan_lab.data import load_prebuilt_windows
        tensors = load_prebuilt_windows(windows_dir, date_range or [])
    else:
        tensors = build_sequences(df, factor_columns, timestep=timestep)
    dataset = TensorDataset(tensors.z, tensors.y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _gradient_penalty(
    model: FactorGAN,
    z: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
    gp_lambda: float,
) -> torch.Tensor:
    """WGAN-GP 梯度惩罚项（适配 D 方式 B：对 [h_last, r] 空间求梯度）。"""
    batch_size = z.size(0)
    alpha = torch.rand(batch_size, 1, device=z.device)
    # 给 real/fake 扩一维，避免 (B,1)*(B,) → (B,B) 的广播问题
    r_interp = (alpha * real.unsqueeze(-1) + (1 - alpha) * fake.unsqueeze(-1)).squeeze(-1)  # (B,)

    # LSTM 编码纯因子（保持梯度图）
    h = model.D.lstm(z)[0]                     # (B, T, hidden)
    h_last = h[:, -1, :]                        # (B, hidden)
    combined = torch.cat([h_last, r_interp.unsqueeze(-1)], dim=-1)  # (B, hidden+1)

    grad_outputs = torch.ones(batch_size, device=z.device)
    gradients = autograd.grad(
        outputs=model.D.head(combined).squeeze(-1),
        inputs=combined,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return gp_lambda * ((gradient_norm - 1) ** 2).mean()


@torch.no_grad()
def eval_val_mse(model: FactorGAN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for z, y in loader:
        z = z.to(device)
        y = y.to(device)
        r_hat = model.G(z)
        loss = torch.mean((r_hat - y) ** 2)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_metrics(model: FactorGAN, loader: DataLoader, device: torch.device) -> dict:
    """计算验证集上的 IC、ICIR、方向胜率、MSE。"""
    from scipy.stats import spearmanr

    model.eval()
    all_y: list[float] = []
    all_pred: list[float] = []
    mse_list: list[float] = []

    for z, y in loader:
        z = z.to(device)
        y = y.to(device)
        r_hat = model.G(z)
        mse_list.append(float(torch.mean((r_hat - y) ** 2).item()))
        all_y.extend(y.cpu().tolist())
        all_pred.extend(r_hat.cpu().tolist())

    y_arr = np.array(all_y)
    pred_arr = np.array(all_pred)

    ic, _ = spearmanr(pred_arr, y_arr) if len(y_arr) > 1 else (0.0, 1.0)
    mse_val = float(np.mean(mse_list)) if mse_list else float("nan")
    direction_hit = float(np.mean((np.sign(pred_arr) == np.sign(y_arr))))

    return {"ic": ic, "mse": mse_val, "direction_hit": direction_hit}


def train_one_window(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    factor_columns: list[str],
    cfg: FactorGANConfig,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    n_critic: int,
    mse_weight: float = 10.0,
    windows_dir: str | None = None,
    train_period: list | None = None,
    val_period: list | None = None,
) -> tuple[FactorGAN, dict]:
    """训练单个滚动窗口（WGAN-GP）。"""
    model = FactorGAN(cfg).to(device)
    opt_g = torch.optim.Adam(model.G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(model.D.parameters(), lr=lr, betas=(0.5, 0.9))

    train_loader = _build_loader(df_train, factor_columns=factor_columns, batch_size=batch_size, shuffle=True, timestep=cfg.timestep, windows_dir=windows_dir, date_range=train_period)
    val_loader = _build_loader(df_val, factor_columns=factor_columns, batch_size=batch_size, shuffle=False, timestep=cfg.timestep, windows_dir=windows_dir, date_range=val_period)

    best_val = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        d_losses: list[float] = []
        g_losses: list[float] = []
        adv_losses: list[float] = []
        mse_losses: list[float] = []
        gp_losses: list[float] = []

        for z, y in train_loader:
            z = z.to(device)
            y = y.to(device)

            # ---- 判别器更新 (n_critic 次) ----
            for _ in range(n_critic):
                r_hat = model.G(z).detach()
                d_real = model.D(z, y)     # 方式 B: LSTM(z) → h_last + y
                d_fake = model.D(z, r_hat) # 方式 B: LSTM(z) → h_last + r_hat
                gp = _gradient_penalty(model, z, y, r_hat, cfg.gp_lambda)
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake)) + gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                d_losses.append(float(d_loss.item()))
                gp_losses.append(float(gp.item()))

            # ---- 生成器更新 ----
            r_hat = model.G(z)
            d_fake = model.D(z, r_hat)     # 方式 B
            adv_loss = -torch.mean(d_fake)
            mse = torch.mean((r_hat - y) ** 2)
            g_loss = adv_loss + mse_weight * mse

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_losses.append(float(g_loss.item()))
            adv_losses.append(float(adv_loss.item()))
            mse_losses.append(float(mse.item()))

        val_metrics = eval_metrics(model, val_loader, device=device)
        val_mse = val_metrics["mse"]
        epoch_row = {
            "epoch": epoch,
            "d_loss": float(np.mean(d_losses)) if d_losses else float("nan"),
            "g_loss": float(np.mean(g_losses)) if g_losses else float("nan"),
            "adv_loss": float(np.mean(adv_losses)) if adv_losses else float("nan"),
            "train_mse": float(np.mean(mse_losses)) if mse_losses else float("nan"),
            "gp": float(np.mean(gp_losses)) if gp_losses else float("nan"),
            "val_mse": float(val_mse),
            "val_ic": float(val_metrics["ic"]),
            "val_direction_hit": float(val_metrics["direction_hit"]),
        }
        history.append(epoch_row)

        print(
            f"  Epoch {epoch:3d} | "
            f"D_loss={epoch_row['d_loss']:.4f} "
            f"G_loss={epoch_row['g_loss']:.4f} "
            f"Adv={epoch_row['adv_loss']:.4f} "
            f"MSE={epoch_row['train_mse']:.6f} "
            f"GP={epoch_row['gp']:.4f} | "
            f"val_mse={val_mse:.6f} "
            f"val_ic={val_metrics['ic']:.4f} "
            f"dir_hit={val_metrics['direction_hit']:.3f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_state = {
                "G": {key: value.detach().cpu() for key, value in model.G.state_dict().items()},
                "D": {key: value.detach().cpu() for key, value in model.D.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"  早停于 epoch {epoch}, best_val_mse={best_val:.6f}")
            break

    if best_state is not None:
        model.G.load_state_dict(best_state["G"])
        model.D.load_state_dict(best_state["D"])

    info = {
        "best_val_mse": best_val,
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "history": history,
    }
    return model, info


def _compute_test_metrics(pred: pd.DataFrame) -> dict:
    """从预测结果计算 IC、ICIR、方向胜率。"""
    from scipy.stats import spearmanr

    if pred.empty or "y" not in pred.columns or "y_pred" not in pred.columns:
        return {"ic": 0.0, "icir": 0.0, "direction_hit": 0.5}

    daily_ics: list[float] = []
    for _, day_group in pred.groupby("trade_date", sort=False):
        if len(day_group) < 5:
            continue
        ic, _ = spearmanr(day_group["y_pred"], day_group["y"])
        if not np.isnan(ic):
            daily_ics.append(ic)

    ic = float(np.mean(daily_ics)) if daily_ics else 0.0
    ic_std = float(np.std(daily_ics)) if len(daily_ics) > 1 else 1.0
    icir = ic / ic_std if ic_std > 0 else 0.0
    direction_hit = float(np.mean(np.sign(pred["y_pred"]) == np.sign(pred["y"])))

    return {"ic": ic, "icir": icir, "direction_hit": direction_hit}


@torch.no_grad()
def predict_window(
    model: FactorGAN,
    df_test: pd.DataFrame,
    factor_columns: list[str],
    device: torch.device,
    timestep: int = 5,
) -> pd.DataFrame:
    """对测试期数据做预测，输出每只股票每日的预测收益。"""
    model.eval()
    tensors = build_sequences(df_test, factor_columns, timestep=timestep)
    if len(tensors.z) == 0:
        return pd.DataFrame(columns=["ts_code", "trade_date", "y", "y_pred"])

    z = tensors.z.to(device)
    r_hat = model.G(z).detach().cpu().numpy()
    out = tensors.meta.copy()
    out["y_pred"] = r_hat
    return out


def run_rolling_training(
    frame: pd.DataFrame,
    factor_columns: list[str],
    result_dir: str | Path,
    device: torch.device,
    train_days: int = 504,
    val_days: int = 63,
    test_days: int = 21,
    step_days: int = 21,
    max_epochs: int = 200,
    patience: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    n_critic: int = 10,
    mse_weight: float = 10.0,
    max_windows: int = 0,
    windows_dir: str | None = None,
) -> None:
    """
    日频滚动窗口训练。

    数据切分逻辑：
    - 训练集：过去 train_days 个交易日
    - 验证集：之后 val_days 个交易日
    - 测试集：再之后 test_days 个交易日
    - 每次向前滑动 step_days 个交易日
    """
    out_root = Path(result_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dates = sorted(frame["trade_date"].unique().tolist())
    win_size = train_days + val_days + test_days
    if len(dates) < win_size:
        raise ValueError(f"交易日数量不足：至少需要 {win_size} 天，当前只有 {len(dates)} 天。")

    cfg = FactorGANConfig(n_factors=len(factor_columns))

    run_summary: list[dict] = []
    num_windows = (len(dates) - win_size) // step_days + 1
    if int(max_windows) > 0:
        num_windows = min(num_windows, int(max_windows))

    for idx in range(num_windows):
        start = idx * step_days
        train_period = dates[start : start + train_days]
        val_period = dates[start + train_days : start + train_days + val_days]
        test_period = dates[start + train_days + val_days : start + win_size]

        df_train = frame[frame["trade_date"].isin(train_period)]
        df_val = frame[frame["trade_date"].isin(val_period)]
        df_test = frame[frame["trade_date"].isin(test_period)]

        tag = (
            f"train_{train_period[0].date()}_{train_period[-1].date()}"
            f"__val_{val_period[0].date()}_{val_period[-1].date()}"
            f"__test_{test_period[0].date()}_{test_period[-1].date()}"
        ).replace("-", "")
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Window {idx + 1}/{num_windows}: {tag}")
        print(f"{'='*60}")

        model, info = train_one_window(
            df_train=df_train,
            df_val=df_val,
            factor_columns=factor_columns,
            cfg=cfg,
            device=device,
            max_epochs=int(max_epochs),
            patience=int(patience),
            batch_size=int(batch_size),
            lr=float(lr),
            n_critic=int(n_critic),
            mse_weight=float(mse_weight),
            windows_dir=windows_dir,
            train_period=train_period,
            val_period=val_period,
        )

        if windows_dir is not None:
            # prebuilt 模式: 分批预测避免 OOM
            from factor_gan_lab.data import load_prebuilt_windows
            test_tensors = load_prebuilt_windows(windows_dir, test_period)
            model.eval()
            all_r_hat: list[np.ndarray] = []
            z_test = test_tensors.z
            for i in range(0, len(z_test), batch_size):
                batch = z_test[i:i + batch_size].to(device)
                with torch.no_grad():
                    all_r_hat.append(model.G(batch).detach().cpu().numpy())
            r_hat = np.concatenate(all_r_hat, axis=0)
            pred = test_tensors.meta.copy()
            pred["y_pred"] = r_hat
        else:
            pred = predict_window(model, df_test, factor_columns=factor_columns, device=device)
        pred.to_csv(out_dir / "test_pred.csv", index=False, encoding="utf-8-sig")

        # 测试集逐日 IC / ICIR / 方向胜率
        test_metrics = _compute_test_metrics(pred)
        print(
            f"  Test: IC={test_metrics['ic']:.4f}  "
            f"ICIR={test_metrics['icir']:.4f}  "
            f"dir_hit={test_metrics['direction_hit']:.3f}"
        )

        with open(out_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)

        torch.save(
            {
                "cfg": asdict(cfg),
                "factor_columns": factor_columns,
                "G": model.G.state_dict(),
                "D": model.D.state_dict(),
            },
            out_dir / "model.pt",
        )

        run_summary.append(
            {
                "window": tag,
                "device": str(device),
                "n_factors": len(factor_columns),
                "factor_columns": ",".join(factor_columns),
                "best_val_mse": float(info["best_val_mse"]),
                "epochs_ran": int(info["epochs_ran"]),
                "test_ic": float(test_metrics["ic"]),
                "test_icir": float(test_metrics["icir"]),
                "test_direction_hit": float(test_metrics["direction_hit"]),
            }
        )

    pd.DataFrame(run_summary).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
