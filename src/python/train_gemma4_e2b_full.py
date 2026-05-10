#!/usr/bin/env python3

import argparse
from typing import Any

from train import load_config
from train_gemma4_full import train_gemma4_full_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burstchester Gemma 4 E2B full fine-tuning runner")
    parser.add_argument("--config", required=True, help="Path to the training config JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    config["modelRepo"] = str(config.get("modelRepo") or "google/gemma-4-E2B")
    config["trainingMethod"] = "full"
    train_gemma4_full_from_config(config)


if __name__ == "__main__":
    main()
