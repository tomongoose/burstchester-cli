#!/usr/bin/env python3

import argparse
from typing import Any

from train import load_config, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burstchester Gemma 2B IT LoRA fine-tuning runner")
    parser.add_argument("--config", required=True, help="Path to the training config JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    config["modelRepo"] = str(config.get("modelRepo") or "google/gemma-2b-it")
    config["trainingMethod"] = "lora"
    config["maxSeqLength"] = int(config.get("maxSeqLength", 128))
    config["loraRank"] = int(config.get("loraRank", 8))
    config["loraAlpha"] = int(config.get("loraAlpha", 16))
    config["loraDropout"] = float(config.get("loraDropout", 0.05))
    train_from_config(config)


if __name__ == "__main__":
    main()
