"""Evaluation entry point for OIGA."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--num_test_samples", type=int, default=500)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
