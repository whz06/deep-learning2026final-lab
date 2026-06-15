#!/bin/bash
#SBATCH --job-name=factor_gan
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

# ============================================================
# 配置区（按需修改）
# ============================================================
PROJECT_DIR="/home/scc/pb24030836/src/code/final_lab"

# 预构建的 .pt 文件目录（请先上传 windows_v7.zip 并解压）
WINDOWS_DIR="${PROJECT_DIR}/processed/windows_v7"
RESULT_DIR="${PROJECT_DIR}/outputs/experiments"

# ============================================================

cd "${PROJECT_DIR}"
mkdir -p logs "${RESULT_DIR}"

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ai25
export PYTHONUNBUFFERED=1
set -u

echo "===== 训练 (prebuilt 模式, 最后三年) ====="
python scripts/train_factor_gan.py \
    --data_mode prebuilt \
    --windows_dir "${WINDOWS_DIR}" \
    --result_dir "${RESULT_DIR}" \
    --train_days 756 \
    --val_days 126 \
    --test_days 42 \
    --step_days 756 \
    --max_epochs 200 \
    --patience 20 \
    --batch_size 128 \
    --lr 3e-5 \
    --n_critic 3 \
    --mse_weight 0.5 \
    --max_windows 0 \
    --device auto

echo "===== DONE ====="
