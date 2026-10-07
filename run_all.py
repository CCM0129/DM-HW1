"""一行重現組員 3 的所有 CV 結果(不含測試集):python run_all.py
測試集評估另外手動執行 python src/final_test.py --confirm(只跑一次)。"""
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
for script in ["evaluate.py", "baselines.py", "models.py", "collinearity.py", "compare.py"]:
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, str(SRC / script)], check=True, env={**os.environ, "PYTHONUTF8": "1"})
