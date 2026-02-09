#!/usr/bin/env python3
"""Resample prediction to original image geometry."""

import argparse

from spleen_seg.inference.post_process import resample_to_original


def main():
    parser = argparse.ArgumentParser(description="Resample prediction to original image space")
    parser.add_argument("image_orig", help="Original NIfTI image path")
    parser.add_argument("pred", help="Prediction NIfTI path")
    parser.add_argument("output", help="Output NIfTI path")
    args = parser.parse_args()
    resample_to_original(args.image_orig, args.pred, args.output)


if __name__ == "__main__":
    main()
