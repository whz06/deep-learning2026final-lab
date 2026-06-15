"""score_all.py — 按顺序运行所有模型的打分脚本。

用法 (Windows PowerShell):
  & python.exe score_all.py --start 20260105 --end 20260529 --device cuda

每个模型独立打分，中途失败不影响其他脚本。
"""
import subprocess, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    # 文件名              模型名       状态
    ("score_v7spatial.py", "v7spatial", "⏳"),   # V7 GRU+SpatialAttention T+1
    # 后续可扩展:
    # ("score_v6.py",      "v6",        ""),     # V6 Spatial T+5
    # ("score_v7gru.py",   "v7gru",     ""),     # V7 GRU T+1
    # ("score_v8.py",      "v8",        ""),     # V8 Multi-source
]

def main():
    args = sys.argv[1:]
    python = sys.executable

    for script, name, _ in SCRIPTS:
        path = os.path.join(SCRIPT_DIR, script)
        if not os.path.exists(path):
            print(f"[skip] {name} — {script} not found")
            continue

        print(f"\n{'='*60}")
        print(f"[score_all] Running {name} ({script}) ...")
        print(f"{'='*60}\n")
        cmd = [python, path] + args
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[score_all] {name} FAILED (exit={result.returncode})")
        else:
            print(f"\n[score_all] {name} OK")
    print("\n[score_all] All done.")


if __name__ == "__main__":
    main()
