#!/usr/bin/env python3

from pathlib import Path
from typing import Any

from train import (
    SupervisedCollator,
    load_config,
    read_jsonl_messages,
    render_messages,
)


def import_gemma4_training_stack():
    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Gemma 4 full fine-tuning requires a transformers build with "
            "AutoModelForImageTextToText. Install/upgrade: "
            "python -m pip install -U torch accelerate transformers"
        ) from error

    return {
        "torch": torch,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def train_gemma4_full_from_config(config: dict[str, Any]) -> None:
    stack = import_gemma4_training_stack()
    torch = stack["torch"]
    AutoModelForImageTextToText = stack["AutoModelForImageTextToText"]
    AutoProcessor = stack["AutoProcessor"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]

    model_repo = str(config.get("modelRepo") or "google/gemma-4-E2B")
    dataset_path = Path(config["datasetPath"]).resolve()
    output_dir = Path(config["outputDir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True)
    tokenizer = get_processor_tokenizer(processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model_kwargs = resolve_gemma4_model_load_kwargs(torch)
    model = load_gemma4_model(
        AutoModelForImageTextToText,
        model_repo,
        model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_dataset = Gemma4TextSupervisedDataset(
        processor=processor,
        dataset_path=dataset_path,
        max_seq_length=int(config["maxSeqLength"]),
        torch_module=torch,
    )
    collator = SupervisedCollator(tokenizer, torch)

    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    training_args = TrainingArguments(
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
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
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
    processor.save_pretrained(str(output_dir))


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


def resolve_gemma4_model_load_kwargs(torch_module) -> dict[str, Any]:
    return {
        "dtype": torch_module.bfloat16
        if bool(torch_module.cuda.is_available() and getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
        else torch_module.float16
        if torch_module.cuda.is_available()
        else torch_module.float32,
    }


def load_gemma4_model(model_class, model_repo: str, model_kwargs: dict[str, Any]):
    try:
        return model_class.from_pretrained(
            model_repo,
            trust_remote_code=True,
            **model_kwargs,
        )
    except TypeError as error:
        if "dtype" not in model_kwargs:
            raise

        legacy_kwargs = dict(model_kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        try:
            return model_class.from_pretrained(
                model_repo,
                trust_remote_code=True,
                **legacy_kwargs,
            )
        except TypeError:
            raise error


def get_processor_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise SystemExit("Gemma 4 processor does not expose a tokenizer.")
    return tokenizer


def render_gemma4_messages_for_text_only(processor, messages: list[dict[str, str]]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        tokenizer = get_processor_tokenizer(processor)
        return render_messages(tokenizer, messages)


class Gemma4TextSupervisedDataset:
    def __init__(self, processor, dataset_path: Path, max_seq_length: int, torch_module):
        tokenizer = get_processor_tokenizer(processor)
        self._torch = torch_module
        self.samples = []

        for messages in read_jsonl_messages(dataset_path):
            text = render_gemma4_messages_for_text_only(processor, messages)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            input_ids = self._torch.tensor(encoded["input_ids"], dtype=self._torch.long)
            attention_mask = self._torch.tensor(encoded["attention_mask"], dtype=self._torch.long)
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }
            )

        if not self.samples:
            raise SystemExit(f"No training samples found in {dataset_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


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
