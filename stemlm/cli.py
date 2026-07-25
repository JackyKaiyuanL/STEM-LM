import argparse

from stemlm import __version__
from stemlm.train import add_train_args


def main():
    parser = argparse.ArgumentParser(prog="stemlm", description="STEM-LM")
    parser.add_argument("--version", action="version", version=f"stemlm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    add_train_args(sub.add_parser("train", help="Train STEM-LM"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
