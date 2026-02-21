#!/usr/bin/env python3
"""
Data Preparation Script
=======================
Prepares SurfPro dataset for training.
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import argparse

# TODO: Implement data preparation pipeline


def main():
    parser = argparse.ArgumentParser(description="Prepare SurfPro data")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    print("Data preparation script")
    print(f"Raw data: {args.raw_dir}")
    print(f"Output: {args.output_dir}")
    
    # TODO: Implement


if __name__ == "__main__":
    main()
