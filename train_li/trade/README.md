# trade/ — Competition Daily Inference

Daily buy/sell stock ranking with Strategy B momentum stop-loss protection.

## Usage

```bash
# Windows (from WSL)
powershell.exe -Command "& D:\Software\miniconda3\envs\dl_lab1\python.exe D:\Workspace\DL_HW\LAB5\trade\infer.py --date 20260529"

# Dry run (no file writes, just show decision)
powershell.exe -Command "& ... --date 20260529 --dry-run"

# Custom buy/sell count
powershell.exe -Command "& ... --date 20260529 --top-n 30 --bottom-k 10"
```

## Strategy B (Auto-Applied)

```
CSI300 5-day return < -1.0%  →  risk-off, position = 80%
CSI300 5-day return ≥ -1.0%  →  full position, position = 100%
```

Risk-off reduces buy/sell count proportionally:
- buy_n = ceil(top_n * 0.80)
- sell_k = floor(bottom_k * 0.80)

## Output Files

| File | Description |
|------|-------------|
| `buy_list.txt` | Stock codes, one per line, for next-day buy |
| `sell_list.txt` | Stock codes, one per line, for next-day sell |
| `decision.log` | Daily decision log: date, position, csi5d, trigger |

## Daily Workflow

1. After market close (~18:00), updated daily data arrives
2. Run `shared/preprocess.py` to rebuild `processed/all_data.parquet`
3. Run `trade/infer.py --date <latest_date>`
4. Submit `buy_list.txt` / `sell_list.txt` for next trading day

## Reference

- **Model**: GRU H=128 L=1 D=0.2 (v2 best checkpoint)
- **Data**: `processed/all_data.parquet` (10 raw + 8 tech + 4 cross features)
- **CSI300**: `data/market/000300.SH.csv`
