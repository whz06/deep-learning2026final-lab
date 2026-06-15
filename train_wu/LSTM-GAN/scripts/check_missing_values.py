from __future__ import annotations

"""
批量检查 CSV 文件缺失值情况的脚本。

这个脚本保留为一个通用数据质检工具，和具体因子定义无关。
你后续无论换哪一套因子，都可以先用它检查原始表或中间表的缺失情况。
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="扫描单个 CSV 或目录下全部 CSV 的缺失值情况。")
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "data"),
        help="输入文件或目录路径；如果是目录，会递归扫描其中所有 CSV。",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(project_root / "reports" / "missing_report.csv"),
        help="输出缺失汇总报告路径；留空则只在终端打印。",
    )
    return parser.parse_args(argv)


def iter_csv_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"路径不存在：{path}")
    return sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file())


def load_csv(path: Path) -> pd.DataFrame:
    """
    按字符串读取 CSV，尽量保留原始字段状态。

    这样做的原因是：
    - 有些代码、日期、证券代码列不应该被自动转换成整数或浮点数
    - 我们这里只做缺失检查，不做业务含义推断
    """

    return pd.read_csv(
        path,
        dtype="string",
        na_values=["", "NaN", "nan"],
        keep_default_na=True,
    )


def build_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    missing_count = df.isna().sum()
    missing_ratio = missing_count / total if total > 0 else 0.0
    return (
        pd.DataFrame(
            {
                "column": missing_count.index,
                "missing_count": missing_count.to_numpy(dtype=int),
                "missing_ratio": (missing_ratio.to_numpy() * 100.0).round(4),
            }
        )
        .sort_values(["missing_ratio", "missing_count"], ascending=False)
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    csv_paths = iter_csv_paths(args.input)
    if not csv_paths:
        raise FileNotFoundError("未找到任何 CSV 文件。")

    report_rows: list[dict] = []
    files_with_missing = 0

    for csv_path in csv_paths:
        table = build_missing_table(load_csv(csv_path))
        missing_only = table[table["missing_count"] > 0].copy()
        if missing_only.empty:
            continue

        files_with_missing += 1
        print(f"[存在缺失] {csv_path}")
        print(missing_only)

        for row in missing_only.itertuples(index=False):
            report_rows.append(
                {
                    "file": str(csv_path),
                    "column": str(row.column),
                    "missing_count": int(row.missing_count),
                    "missing_ratio": float(row.missing_ratio),
                }
            )

    print(f"扫描完成：共扫描 {len(csv_paths)} 个 CSV，其中 {files_with_missing} 个存在缺失值。")

    report = (args.report or "").strip()
    if report:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report_rows).to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"已生成报告：{report_path}")


if __name__ == "__main__":
    main()
