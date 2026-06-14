from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import RESULT_DIR, TrainConfig
from .data import PriceWindowDataset, WindowArrays, inverse_maxmin, inverse_sigmoid_formula
from .metrics import daily_ic_summary, direction_win_rate_by_period, fill_missing_with_mean
from .models import build_model, pick_device

LOG_EVERY_BATCHES = 500


def build_loader(arrays: WindowArrays, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = PriceWindowDataset(arrays)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def train_one_model(
    cfg: TrainConfig,
    train_arrays: WindowArrays,
    val_arrays: WindowArrays,
    test_arrays: WindowArrays,
    device_name: str = "auto",
) -> dict[str, object]:
    """训练单个模型，并返回测试指标与信号。"""

    device = pick_device(device_name)
    model = build_model(
        model_name=cfg.model_name,
        input_dim=int(train_arrays.x.shape[-1]),
        seq_len=cfg.seq_len,
        hidden=cfg.lstm_hidden,
        filters=cfg.cnn_filters,
        dropout=cfg.dropout,
    ).to(device)

    train_loader = build_loader(train_arrays, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = build_loader(val_arrays, cfg.batch_size, cfg.num_workers, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    global_step = 0
    history = {
        "epoch": [],
        "batch_in_epoch": [],
        "global_step": [],
        "train_loss": [],
        "val_loss": [],
        "test_ic": [],
        "test_ir": [],
        "test_rank_ic": [],
        "test_rank_ir": [],
        "test_direction_win_rate": [],
    }

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses_window: list[float] = []
        num_batches = len(train_loader)
        for batch_idx, (x, y_scaled, *_) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y_scaled = y_scaled.to(device, non_blocking=True)
            pred = model(x)
            loss = torch.mean((pred - y_scaled) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            train_losses_window.append(float(loss.item()))

            should_log = (batch_idx % LOG_EVERY_BATCHES == 0) or (batch_idx == num_batches)
            if not should_log:
                continue

            train_loss = float(np.mean(train_losses_window))
            train_losses_window = []
            val_loss = _eval_scaled_mse(model, val_loader, device)
            eval_result = _evaluate_period_metrics(model, test_arrays, cfg.batch_size, device, cfg.scale_name)
            _append_eval_to_history(
                history,
                metrics=eval_result["metrics"],
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            print(
                f"[{cfg.scale_name}_{cfg.model_name}_L{cfg.seq_len}] "
                f"epoch={epoch:02d} batch={batch_idx:04d}/{num_batches:04d} step={global_step:06d} "
                f"train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
                f"ic={eval_result['metrics']['ic']:.6f} "
                f"ir={eval_result['metrics']['ir']:.6f} "
                f"rank_ic={eval_result['metrics']['rank_ic']:.6f} "
                f"rank_ir={eval_result['metrics']['rank_ir']:.6f} "
                f"dir_win={eval_result['metrics']['direction_win_rate']:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(
                        f"[{cfg.scale_name}_{cfg.model_name}_L{cfg.seq_len}] "
                        f"早停，best_val_mse={best_val:.6f}，触发位置 epoch={epoch} batch={batch_idx}"
                    )
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    final_eval = _evaluate_period_metrics(model, test_arrays, cfg.batch_size, device, cfg.scale_name)
                    return {
                        "model": model,
                        "history": history,
                        "metrics": final_eval["metrics"],
                        "signals": final_eval["signals"],
                        "period_ic": final_eval["period_ic"],
                        "period_direction_metrics": final_eval["period_direction_metrics"],
                        "device": str(device),
                    }

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval = _evaluate_period_metrics(model, test_arrays, cfg.batch_size, device, cfg.scale_name)
    return {
        "model": model,
        "history": history,
        "metrics": final_eval["metrics"],
        "signals": final_eval["signals"],
        "period_ic": final_eval["period_ic"],
        "period_direction_metrics": final_eval["period_direction_metrics"],
        "device": str(device),
    }


@torch.no_grad()
def predict_arrays(
    model: torch.nn.Module,
    arrays: WindowArrays,
    batch_size: int,
    device: torch.device,
    scale_name: str,
) -> pd.DataFrame:
    loader = build_loader(arrays, batch_size=batch_size, num_workers=0, shuffle=False)
    preds_scaled: list[np.ndarray] = []
    was_training = bool(model.training)
    model.eval()
    for x, *_ in loader:
        x = x.to(device, non_blocking=True)
        preds_scaled.append(model(x).detach().cpu().numpy())
    if was_training:
        model.train()

    pred_scaled = np.concatenate(preds_scaled, axis=0).astype(np.float32, copy=False)
    if scale_name == "maxmin":
        pred_raw = inverse_maxmin(pred_scaled, arrays.min_ref, arrays.max_ref)
    elif scale_name == "sigmoid":
        pred_raw = inverse_sigmoid_formula(pred_scaled)
    else:
        raise ValueError(f"未知缩放类型: {scale_name}")
    pred_ret = pred_raw / arrays.cur_close - 1.0
    real_ret = arrays.next_close / arrays.cur_close - 1.0

    return pd.DataFrame(
        {
            "ts_code": arrays.ts_code,
            "signal_date": arrays.signal_date,
            "target_date": arrays.target_date,
            "cur_close": arrays.cur_close,
            "next_close": arrays.next_close,
            "y_scaled": arrays.y_scaled,
            "pred_scaled": pred_scaled,
            "y_raw": arrays.y_raw,
            "pred_raw": pred_raw,
            "pred_ret": pred_ret,
            "real_ret": real_ret,
        }
    )


def _evaluate_period_metrics(
    model: torch.nn.Module,
    test_arrays: WindowArrays,
    batch_size: int,
    device: torch.device,
    scale_name: str,
) -> dict[str, object]:
    signals = predict_arrays(model, test_arrays, batch_size, device, scale_name)
    signals = fill_missing_with_mean(
        signals,
        ["cur_close", "next_close", "y_scaled", "pred_scaled", "y_raw", "pred_raw", "pred_ret", "real_ret"],
    )
    period_ic, ic_summary = daily_ic_summary(signals)
    direction_period, direction_summary = direction_win_rate_by_period(signals)
    metrics = {
        "ic": ic_summary["ic"],
        "ir": ic_summary["ir"],
        "rank_ic": ic_summary["rank_ic"],
        "rank_ir": ic_summary["rank_ir"],
        "direction_win_rate": direction_summary["direction_win_rate"],
    }
    return {
        "signals": signals,
        "period_ic": period_ic,
        "period_direction_metrics": direction_period,
        "metrics": metrics,
    }


def _append_eval_to_history(
    history: dict[str, list[float]],
    metrics: dict[str, object],
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
) -> None:
    history["epoch"].append(int(epoch))
    history["batch_in_epoch"].append(int(batch_in_epoch))
    history["global_step"].append(int(global_step))
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["test_ic"].append(float(metrics["ic"]))
    history["test_ir"].append(float(metrics["ir"]))
    history["test_rank_ic"].append(float(metrics["rank_ic"]))
    history["test_rank_ir"].append(float(metrics["rank_ir"]))
    history["test_direction_win_rate"].append(float(metrics["direction_win_rate"]))


def save_train_result(result: dict[str, object], cfg: TrainConfig, result_dir: Path | None = None) -> dict[str, Path]:
    result_dir = RESULT_DIR if result_dir is None else result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{cfg.scale_name}_{cfg.model_name}_L{cfg.seq_len}"

    ckpt_path = result_dir / f"{run_tag}.pt"
    metrics_path = result_dir / f"{run_tag}.json"
    signals_path = result_dir / f"{run_tag}_test_signals.csv.gz"
    ic_path = result_dir / f"{run_tag}_period_ic.csv"
    direction_path = result_dir / f"{run_tag}_period_direction_metrics.csv"

    torch.save(
        {
            "model_cfg": asdict(cfg),
            "state_dict": result["model"].state_dict(),
            "seq_len": cfg.seq_len,
        },
        ckpt_path,
    )
    result["signals"].to_csv(signals_path, index=False, encoding="utf-8")
    result["period_ic"].to_csv(ic_path, index=False, encoding="utf-8")
    result["period_direction_metrics"].to_csv(direction_path, index=False, encoding="utf-8")

    payload = {
        "model_cfg": asdict(cfg),
        "metrics": result["metrics"],
        "history": result["history"],
        "device": result["device"],
        "signals_path": str(signals_path),
        "period_ic_path": str(ic_path),
        "period_direction_metrics_path": str(direction_path),
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ckpt": ckpt_path,
        "metrics": metrics_path,
        "signals": signals_path,
        "period_ic": ic_path,
        "period_direction_metrics": direction_path,
    }


@torch.no_grad()
def load_checkpoint(model_path: Path, device_name: str = "auto") -> tuple[torch.nn.Module, TrainConfig, torch.device]:
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["model_cfg"])
    device = pick_device(device_name)
    model = build_model(
        model_name=cfg.model_name,
        input_dim=1,
        seq_len=cfg.seq_len,
        hidden=cfg.lstm_hidden,
        filters=cfg.cnn_filters,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg, device


@torch.no_grad()
def _eval_scaled_mse(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    was_training = bool(model.training)
    model.eval()
    losses: list[float] = []
    for x, y_scaled, *_ in loader:
        x = x.to(device, non_blocking=True)
        y_scaled = y_scaled.to(device, non_blocking=True)
        pred = model(x)
        losses.append(float(torch.mean((pred - y_scaled) ** 2).item()))
    if was_training:
        model.train()
    return float(np.mean(losses))
