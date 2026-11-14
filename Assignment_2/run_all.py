##########################################################################
#   Run the A2.2 experiments one after another (each in its own          #
#   process). Finished runs are skipped and an interrupted run resumes   #
#   from its last checkpoint, so the command can simply be rerun after   #
#   a Colab disconnect. --arch lets two Colab accounts split the work.   #
#   Example: python run_all.py --arch segformer_b2 --run-dir <Drive dir> #
##########################################################################

import argparse
import subprocess
import sys
from pathlib import Path

import config

# Main comparison (100% data) first, then the smaller training sets
ORDER = sorted(config.EXPERIMENTS, key = lambda e: -e["train_fraction"])


def main():
    parser = argparse.ArgumentParser(description = "Chạy lần lượt các thí nghiệm A2.2")
    parser.add_argument("--run-dir", default = str(config.OUTPUT_DIR / "runs"))
    parser.add_argument("--arch", choices = list(config.ARCHS), default = None, help = "Chỉ chạy một kiến trúc")
    parser.add_argument("--only", nargs = "*", default = None, help = "Chỉ chạy các cấu hình này")
    parser.add_argument("--epochs", type = int, default = None)
    parser.add_argument("--ckpt-dir", default = None, help = "Thư mục checkpoint chạy tiếp (mặc định = --run-dir)")
    parser.add_argument("--workers", type = int, default = 2)
    args = parser.parse_args()

    for exp in ORDER:
        if (args.arch and exp["arch"] != args.arch) or (args.only and exp["name"] not in args.only):
            continue
        cmd = [sys.executable, "train.py", "--exp", exp["name"], "--run-dir", args.run_dir, "--workers", str(args.workers)]
        if args.ckpt_dir:
            cmd += ["--ckpt-dir", args.ckpt_dir]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print("\n>>>", " ".join(cmd), flush = True)
        subprocess.run(cmd, check = True, cwd = Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
