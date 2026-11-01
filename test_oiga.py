"""Evaluation entry point for OIGA.

The interface defaults to the 500 test images per modality reported in the
paper. The full implementation is not included in this review-stage subset.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--num_test_samples", type=int, default=500)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.parse_args()
    raise NotImplementedError("The full entry point is not included in this subset.")


if __name__ == "__main__":
    main()
