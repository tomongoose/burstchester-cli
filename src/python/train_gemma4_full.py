#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any

from train import (
    load_config,
    read_jsonl_messages,
    render_messages,
)


def import_unsloth_gemma4_training_stack():
    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit(
            "Gemma 4 Unsloth full fine-tuning requires: "
            "unsloth unsloth_zoo datasets trl torch. Install/upgrade: "
            "python -m pip install --upgrade --force-reinstall --no-cache-dir "
            "unsloth unsloth_zoo"
        ) from error

    return {
        "torch": torch,
        "Dataset": Dataset,
        "FastLanguageModel": FastLanguageModel,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def train_gemma4_full_from_config(config: dict[str, Any]) -> None:
    stack = import_unsloth_gemma4_training_stack()
    torch = stack["torch"]
    Dataset = stack["Dataset"]
    FastLanguageModel = stack["FastLanguageModel"]
    SFTConfig = stack["SFTConfig"]
    SFTTrainer = stack["SFTTrainer"]

    model_repo = str(config.get("modelRepo") or "google/gemma-4-E2B")
    dataset_path = Path(config["datasetPath"]).resolve()
    output_dir = Path(config["outputDir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_seq_length = int(config["maxSeqLength"])

    require_cuda_for_gemma4_fft(torch)
    require_supported_fft_gpu(torch)
    log_cuda_memory(torch, "before model load")
    model, tokenizer = load_unsloth_gemma4_model(
        FastLanguageModel,
        model_repo=model_repo,
        max_seq_length=max_seq_length,
    )
    log_model_device(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    if hasattr(model, "config"):
        model.config.use_cache = False

    train_dataset = build_unsloth_text_dataset(
        Dataset,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
    )

    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(config["numTrainEpochs"]),
        per_device_train_batch_size=int(config["perDeviceTrainBatchSize"]),
        gradient_accumulation_steps=int(config["gradientAccumulationSteps"]),
        learning_rate=float(config["learningRate"]),
        logging_steps=int(config["loggingSteps"]),
        save_steps=int(config["saveSteps"]),
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_8bit",
        seed=3407,
        bf16=use_bf16,
        fp16=use_fp16,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    log_cuda_memory(torch, "before training")
    oom_error = getattr(torch, "OutOfMemoryError", RuntimeError)
    try:
        trainer.train()
    except oom_error as error:
        raise SystemExit(
            "CUDA out of memory during Gemma 4 full fine-tuning backward pass. "
            "The model was already loaded; this allocation is for train-time "
            "gradients/activations/optimizer state required by FFT. Use a larger "
            "GPU, reduce maxSeqLength further, or switch to LoRA/QLoRA for Colab-class GPUs."
        ) from error
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def require_cuda_for_gemma4_fft(torch_module) -> None:
    if torch_module.cuda.is_available():
        device = torch_module.cuda.current_device()
        print(
            "CUDA available: "
            f"device={device} name={torch_module.cuda.get_device_name(device)} "
            f"torch={torch_module.__version__} cuda={torch_module.version.cuda}",
            flush=True,
        )
        return

    raise SystemExit(
        "Gemma 4 full fine-tuning requires a CUDA GPU runtime. "
        "In Colab, open Runtime > Change runtime type and select a GPU, "
        "then rerun the notebook."
    )


def require_supported_fft_gpu(torch_module) -> None:
    device = torch_module.cuda.current_device()
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    total_gib = total_bytes / 1024**3
    is_bf16_supported = bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
    allow_low_vram = os.environ.get("BURSTCHESTER_ALLOW_LOW_VRAM_GEMMA4_FFT") == "1"

    if is_bf16_supported and total_gib >= 24:
        return

    if allow_low_vram:
        print(
            "WARNING: overriding Gemma 4 FFT GPU guard. "
            f"bf16={is_bf16_supported} total_vram={total_gib:.2f}GiB",
            flush=True,
        )
        return

    raise SystemExit(
        "Gemma 4 FFT is not supported on this GPU setup. "
        "Unsloth reports that float16 full fine-tuning upcasts weights to float32, "
        "which exceeds Colab T4-class VRAM during model preparation. "
        f"Detected bf16={is_bf16_supported}, total_vram={total_gib:.2f}GiB. "
        "Use an A100/L4-class bf16 GPU with more VRAM, or set "
        "BURSTCHESTER_ALLOW_LOW_VRAM_GEMMA4_FFT=1 to try anyway."
    )


def load_unsloth_gemma4_model(FastLanguageModel, model_repo: str, max_seq_length: int):
    kwargs = {
        "model_name": model_repo,
        "max_seq_length": max_seq_length,
        "dtype": None,
        "load_in_4bit": False,
        "full_finetuning": True,
        "use_gradient_checkpointing": "unsloth",
    }
    try:
        return FastLanguageModel.from_pretrained(**kwargs)
    except TypeError as error:
        legacy_kwargs = dict(kwargs)
        legacy_kwargs.pop("use_gradient_checkpointing")
        try:
            return FastLanguageModel.from_pretrained(**legacy_kwargs)
        except TypeError:
            raise error


def log_model_device(model) -> None:
    try:
        parameter = next(model.parameters())
    except Exception:
        print("Model device: unable to inspect first parameter", flush=True)
        return

    print(
        "Model first parameter: "
        f"device={parameter.device} dtype={parameter.dtype} requires_grad={parameter.requires_grad}",
        flush=True,
    )


def log_cuda_memory(torch_module, label: str) -> None:
    if not torch_module.cuda.is_available():
        return

    device = torch_module.cuda.current_device()
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    allocated_bytes = torch_module.cuda.memory_allocated(device)
    reserved_bytes = torch_module.cuda.memory_reserved(device)
    gib = 1024**3
    print(
        "CUDA memory "
        f"{label}: free={free_bytes / gib:.2f}GiB "
        f"total={total_bytes / gib:.2f}GiB "
        f"allocated={allocated_bytes / gib:.2f}GiB "
        f"reserved={reserved_bytes / gib:.2f}GiB",
        flush=True,
    )


def build_unsloth_text_dataset(Dataset, tokenizer, dataset_path: Path):
    samples = []
    for messages in read_jsonl_messages(dataset_path):
        samples.append({"text": render_gemma4_messages_for_text_only(tokenizer, messages)})

    if not samples:
        raise SystemExit(f"No training samples found in {dataset_path}")

    return Dataset.from_list(samples)


def render_gemma4_messages_for_text_only(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return render_messages(tokenizer, messages)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Burstchester Gemma 4 full fine-tuning runner")
    parser.add_argument("--config", required=True, help="Path to the training config JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    config["modelRepo"] = str(config.get("modelRepo") or "google/gemma-4-E2B")
    config["trainingMethod"] = "full"
    train_gemma4_full_from_config(config)


if __name__ == "__main__":
    main()
