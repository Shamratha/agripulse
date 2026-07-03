import argparse

from agripulse.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AgriPulse crop/stress/irrigation pipeline")
    p.add_argument("--mode", choices=["sample", "gee"], default="sample",
                   help="sample = synthetic pilot area (no accounts needed); gee = Google Earth Engine")
    p.add_argument("--at", type=int, default=None, metavar="T",
                   help="composite index to analyse (0-based; default: last)")
    args = p.parse_args()
    run(args.mode, args.at)
