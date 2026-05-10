#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burstchester local fine-tuning runner")
    parser.add_argument("--config", required=True, help="Path to the training config JSON")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def import_training_stack(training_method: str):
    try:
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Missing Python dependencies. Install at least: torch transformers peft"
        ) from error

    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as error:
        if training_method in {"lora", "qlora"}:
            raise SystemExit("LoRA training requires the `peft` package.") from error
        LoraConfig = None
        get_peft_model = None
        prepare_model_for_kbit_training = None

    return {
        "torch": torch,
        "AutoProcessor": AutoProcessor,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


def read_jsonl_messages(path: Path) -> list[list[dict[str, str]]]:
    samples: list[list[dict[str, str]]] = []
    with open(path, "r", encoding="utf-8") as handle:
      for line in handle:
          line = line.strip()
          if not line:
              continue
          parsed = json.loads(line)
          messages = parsed.get("messages")
          if not isinstance(messages, list):
              raise SystemExit(f"Invalid sample in {path}: missing messages array")
          samples.append(messages)

    if not samples:
        raise SystemExit(f"No training samples found in {path}")
    return samples


def render_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        chunks = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            chunks.append(f"<|{role}|>\n{content}")
        return "\n".join(chunks)


class JsonlSupervisedDataset:
    def __init__(self, tokenizer, dataset_path: Path, max_seq_length: int, torch_module):
        self._torch = torch_module
        self.samples = []

        for messages in read_jsonl_messages(dataset_path):
            text = render_messages(tokenizer, messages)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SupervisedCollator:
    def __init__(self, tokenizer, torch_module):
        self.tokenizer = tokenizer
        self.torch = torch_module

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded_inputs = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        padded_labels = padded_labels.masked_fill(
            padded_inputs["attention_mask"] == 0,
            -100,
        )

        padded_inputs["labels"] = padded_labels
        return padded_inputs


def build_model(config: dict[str, Any], stack: dict[str, Any]):
    torch = stack["torch"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]
    LoraConfig = stack["LoraConfig"]
    get_peft_model = stack["get_peft_model"]
    prepare_model_for_kbit_training = stack["prepare_model_for_kbit_training"]

    training_method = config["trainingMethod"]
    common_kwargs = resolve_model_load_kwargs(
        model_repo=config["modelRepo"],
        training_method=training_method,
        torch_module=torch,
    )

    if training_method == "qlora":
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as error:
            raise SystemExit(
                "QLoRA requires bitsandbytes support. Install bitsandbytes and use a compatible GPU."
            ) from error

        model = AutoModelForCausalLM.from_pretrained(
            config["modelRepo"],
            device_map="auto",
            quantization_config=quantization_config,
            **common_kwargs,
        )
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=int(config["loraRank"]),
            lora_alpha=int(config["loraAlpha"]),
            lora_dropout=float(config["loraDropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolve_lora_target_modules(config["modelRepo"]),
        )
        return get_peft_model(model, peft_config)

    model = AutoModelForCausalLM.from_pretrained(config["modelRepo"], **common_kwargs)
    if training_method == "lora":
        peft_config = LoraConfig(
            r=int(config["loraRank"]),
            lora_alpha=int(config["loraAlpha"]),
            lora_dropout=float(config["loraDropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolve_lora_target_modules(config["modelRepo"]),
        )
        model = get_peft_model(model, peft_config)

    return model


def resolve_lora_target_modules(model_repo: str) -> list[str]:
    if is_gemma_model(model_repo):
        return ["q_proj", "v_proj"]

    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_model_load_kwargs(
    model_repo: str,
    training_method: str,
    torch_module,
) -> dict[str, Any]:
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    if training_method == "qlora":
        return common_kwargs

    if is_gemma_model(model_repo):
        common_kwargs["device_map"] = "auto"
        common_kwargs["torch_dtype"] = (
            torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32
        )

    return common_kwargs


def is_gemma_model(model_repo: str) -> bool:
    return "gemma" in str(model_repo or "").lower()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_from_config(config)


def train_from_config(config: dict[str, Any]) -> None:
    stack = import_training_stack(config["trainingMethod"])

    torch = stack["torch"]
    AutoProcessor = stack["AutoProcessor"]
    AutoTokenizer = stack["AutoTokenizer"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]

    dataset_path = Path(config["datasetPath"]).resolve()
    output_dir = Path(config["outputDir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["modelRepo"], AutoTokenizer, AutoProcessor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = build_model(config, stack)

    train_dataset = JsonlSupervisedDataset(
        tokenizer=tokenizer,
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
        gradient_checkpointing=config["trainingMethod"] in {"lora", "qlora"},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def load_tokenizer(model_repo: str, auto_tokenizer, auto_processor):
    last_error: Exception | None = None

    try:
        processor = auto_processor.from_pretrained(model_repo, trust_remote_code=True)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            return processor.tokenizer
    except Exception as error:
        last_error = error

    try:
        return auto_tokenizer.from_pretrained(model_repo, trust_remote_code=True)
    except Exception as error:
        second_error = error

    try:
        return auto_tokenizer.from_pretrained(
            model_repo,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as error:
        if last_error is not None:
            raise error from last_error
        raise error from second_error


if __name__ == "__main__":
    main()
