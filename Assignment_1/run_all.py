##########################################################################
#   Run all 8 A1.2 experiments one after another (each in its own        #
#   process). Finished runs are skipped and an interrupted run resumes   #
#   from its last checkpoint, so the command can simply be rerun after   #
#   a Colab disconnect.                                                  #
#   Example: python run_all.py --run-dir /content/drive/MyDrive/A1_runs  #
##########################################################################

import argparse
import subprocess
import sys
from pathlib import Path

import config

# Pretrained runs first (10 epochs each, fast), then the two 30-epoch from-scratch runs
ORDER = sorted(config.EXPERIMENTS, key = lambda e: not e["pretrained"])


def main():
    parser = argparse.ArgumentParser(description = "Chạy lần lượt 8 thí nghiệm A1.2")
    parser.add_argument("--run-dir", default = str(config.OUTPUT_DIR / "runs"))
    parser.add_argument("--only", nargs = "*", default = None, help = "Chỉ chạy các cấu hình này")
    parser.add_argument("--workers", type = int, default = 2)
    args = parser.parse_args()

    for exp in ORDER:
        if args.only and exp["name"] not in args.only:
            continue
        cmd = [sys.executable, "train.py", "--exp", exp["name"], "--run-dir", args.run_dir, "--workers", str(args.workers)]
        print("\n>>>", " ".join(cmd), flush = True)
        subprocess.run(cmd, check = True, cwd = Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
