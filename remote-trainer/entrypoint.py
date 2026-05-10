#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DATASET_DOWNLOAD_URL = "https://us-central1-bustchester-e08c3.cloudfunctions.net/prepareDatasetDownload"
DEFAULT_MODEL_REGISTER_URL = "https://us-central1-bustchester-e08c3.cloudfunctions.net/registerModelHttp"
TRAINER_SCRIPTS = {
    "train": "/app/src/python/train.py",
    "train-gemma-2b-it-lora": "/app/src/python/train_gemma_2b_it_lora.py",
    "train-gemma4-e2b-full": "/app/src/python/train_gemma4_e2b_full.py",
}


def main() -> None:
    settings = read_settings()
    workspace = Path(settings["workspace"]).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    dataset_ids = parse_dataset_ids(settings["dataset_ids"])
    merged_dataset_path = download_and_merge_datasets(
        dataset_ids=dataset_ids,
        workspace=workspace,
        endpoint_url=settings["dataset_download_url"],
        access_token=settings["burstchester_access_token"],
    )
    config_path = write_train_config(
        settings=settings,
        dataset_ids=dataset_ids,
        merged_dataset_path=merged_dataset_path,
        workspace=workspace,
    )
    run_trainer(settings["train_command"], config_path)
    upload_to_hugging_face(
        output_dir=workspace / "output",
        repo_id=settings["output_model_repo"],
        hf_token=settings["hf_token"],
    )

    if not settings["skip_register"]:
        register_model(
            endpoint_url=settings["model_register_url"],
            access_token=settings["burstchester_access_token"],
            huggingface_url=f"https://huggingface.co/{settings['output_model_repo']}",
            base_model=settings["base_model"],
            dataset_ids=dataset_ids,
            training_method=effective_training_method(settings["train_command"], settings["training_method"]),
            point_cost=settings["model_point_cost"],
        )


def read_settings() -> dict:
    return {
        "burstchester_access_token": required_env("BURSTCHESTER_ACCESS_TOKEN"),
        "hf_token": required_env("HF_TOKEN"),
        "dataset_ids": required_env("DATASET_IDS"),
        "output_model_repo": required_env("OUTPUT_MODEL_REPO"),
        "base_model": env("BASE_MODEL", "google/gemma-4-E2B"),
        "train_command": env("TRAIN_COMMAND", "train-gemma4-e2b-full"),
        "training_method": env("TRAINING_METHOD", "full"),
        "workspace": env("WORKSPACE", "/workspace/burstchester-training"),
        "epochs": env("EPOCHS", "1"),
        "batch_size": env("BATCH_SIZE", "1"),
        "max_seq_length": env("MAX_SEQ_LENGTH", "128"),
        "grad_accum": env("GRAD_ACCUM", "8"),
        "learning_rate": env("LEARNING_RATE", "0.00005"),
        "lora_rank": env("LORA_RANK", "8"),
        "lora_alpha": env("LORA_ALPHA", "16"),
        "lora_dropout": env("LORA_DROPOUT", "0.05"),
        "logging_steps": env("LOGGING_STEPS", "10"),
        "save_steps": env("SAVE_STEPS", "100"),
        "dataset_download_url": env("DATASET_DOWNLOAD_URL", DEFAULT_DATASET_DOWNLOAD_URL),
        "model_register_url": env("MODEL_REGISTER_URL", DEFAULT_MODEL_REGISTER_URL),
        "model_point_cost": env("MODEL_POINT_COST", "100"),
        "skip_register": env_bool("SKIP_REGISTER", default=True),
    }


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_dataset_ids(raw: str) -> list[str]:
    dataset_ids = list(dict.fromkeys(value for value in re.split(r"[\s,]+", raw.strip()) if value))
    if not dataset_ids:
        raise SystemExit("DATASET_IDS did not contain any dataset ids.")
    return dataset_ids


def download_and_merge_datasets(
    *,
    dataset_ids: list[str],
    workspace: Path,
    endpoint_url: str,
    access_token: str,
) -> Path:
    merged_dataset_path = workspace / "merged-dataset.jsonl"
    dataset_root = workspace / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    with merged_dataset_path.open("w", encoding="utf-8") as merged:
        for dataset_id in dataset_ids:
            metadata = fetch_dataset_metadata(endpoint_url, dataset_id, access_token)
            zip_path = workspace / f"{dataset_id}.zip"
            urllib.request.urlretrieve(metadata["url"], zip_path)

            dataset_dir = dataset_root / dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(dataset_dir)

            jsonl_path = dataset_dir / "dataset.jsonl"
            text = jsonl_path.read_text(encoding="utf-8")
            merged.write(text if text.endswith("\n") else f"{text}\n")

    print(f"Merged datasets {dataset_ids} into {merged_dataset_path}", flush=True)
    return merged_dataset_path


def fetch_dataset_metadata(endpoint_url: str, dataset_id: str, access_token: str) -> dict:
    url = f"{endpoint_url}?{urllib.parse.urlencode({'datasetId': dataset_id})}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not payload.get("ok") or not payload.get("url"):
        raise SystemExit(payload.get("error") or f"Dataset download failed for {dataset_id}")
    return payload


def write_train_config(
    *,
    settings: dict,
    dataset_ids: list[str],
    merged_dataset_path: Path,
    workspace: Path,
) -> Path:
    train_command = settings["train_command"]
    config = {
        "datasetId": dataset_ids[0],
        "datasetIds": dataset_ids,
        "datasetPath": str(merged_dataset_path),
        "modelRepo": settings["base_model"],
        "outputDir": str(workspace / "output"),
        "trainingMethod": effective_training_method(train_command, settings["training_method"]),
        "numTrainEpochs": float(settings["epochs"]),
        "perDeviceTrainBatchSize": int(settings["batch_size"]),
        "gradientAccumulationSteps": int(settings["grad_accum"]),
        "learningRate": float(settings["learning_rate"]),
        "maxSeqLength": int(settings["max_seq_length"]),
        "loraRank": int(settings["lora_rank"]),
        "loraAlpha": int(settings["lora_alpha"]),
        "loraDropout": float(settings["lora_dropout"]),
        "loggingSteps": int(settings["logging_steps"]),
        "saveSteps": int(settings["save_steps"]),
    }
    config_path = workspace / "train-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(config_path.read_text(encoding="utf-8"), flush=True)
    return config_path


def effective_training_method(train_command: str, training_method: str) -> str:
    if train_command == "train-gemma4-e2b-full":
        return "full"
    if train_command == "train-gemma-2b-it-lora":
        return "lora"
    return training_method


def run_trainer(train_command: str, config_path: Path) -> None:
    trainer_script = TRAINER_SCRIPTS.get(train_command)
    if not trainer_script:
        raise SystemExit(f"Unsupported TRAIN_COMMAND: {train_command}")

    command = ["python3", trainer_script, "--config", str(config_path)]
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def upload_to_hugging_face(output_dir: Path, repo_id: str, hf_token: str) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as error:
        raise SystemExit("huggingface_hub is required to upload trained models.") from error

    if not output_dir.exists():
        raise SystemExit(f"Training output directory does not exist: {output_dir}")

    create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="model")
    HfApi(token=hf_token).upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded model to https://huggingface.co/{repo_id}", flush=True)


def register_model(
    *,
    endpoint_url: str,
    access_token: str,
    huggingface_url: str,
    base_model: str,
    dataset_ids: list[str],
    training_method: str,
    point_cost: str,
) -> None:
    body = json.dumps(
        {
            "huggingFaceUrl": huggingface_url,
            "baseModel": base_model,
            "trainingDatasets": dataset_ids,
            "trainingMethod": training_method,
            "pointCost": point_cost,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url,
        data=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        print(response.read().decode("utf-8"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
