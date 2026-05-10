# Burstchester Remote Trainer

This directory packages the Burstchester Python trainers into a GPU Docker image for remote execution on Vertex AI, RunPod, Lambda Labs, or a local CUDA host.

The container reads settings from environment variables, downloads Burstchester datasets, writes the trainer config, runs the selected trainer, uploads the output to Hugging Face, and can optionally register the model back to Burstchester.

## Build

From the `cli` directory:

```bash
docker build -f remote-trainer/Dockerfile -t burstchester-remote-trainer:gemma4 .
```

## Required Environment

```text
BURSTCHESTER_ACCESS_TOKEN
HF_TOKEN
DATASET_IDS
OUTPUT_MODEL_REPO
```

For Vertex AI, prefer Secret Manager resource names instead of raw token values:

```text
BURSTCHESTER_ACCESS_TOKEN_SECRET=projects/PROJECT_ID/secrets/burstchester-access-token/versions/latest
HF_TOKEN_SECRET=projects/PROJECT_ID/secrets/hf-token/versions/latest
```

If both forms are set, the direct token environment variable wins.

## Common Settings

```text
BASE_MODEL=google/gemma-4-E2B
TRAIN_COMMAND=train-gemma4-e2b-full
TRAINING_METHOD=full
WORKSPACE=/workspace/burstchester-training
EPOCHS=1
BATCH_SIZE=1
MAX_SEQ_LENGTH=128
GRAD_ACCUM=8
LEARNING_RATE=0.00005
SKIP_REGISTER=true
MODEL_POINT_COST=100
```

## Local CUDA Run

```bash
docker run --rm --gpus all \
  -e BURSTCHESTER_ACCESS_TOKEN="$BURSTCHESTER_ACCESS_TOKEN" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e DATASET_IDS="legal-ko" \
  -e OUTPUT_MODEL_REPO="your-org/your-gemma4-e2b-fft" \
  -e BASE_MODEL="google/gemma-4-E2B" \
  -e TRAIN_COMMAND="train-gemma4-e2b-full" \
  -e MAX_SEQ_LENGTH="128" \
  -e SKIP_REGISTER="true" \
  burstchester-remote-trainer:gemma4
```

## Vertex AI Notes

Use this image as a Vertex AI Custom Training custom container. Pass user tokens through Secret Manager or job environment variables. Do not bake tokens into the image.

Gemma 4 E2B full fine-tuning is memory-heavy. Use `NVIDIA_A100_80GB`, `NVIDIA_H100_80GB`, or larger for realistic runs. T4/L4/A100 40GB are not expected to complete FFT reliably.

After configuring `gcloud` authentication and project selection, use the scripted launcher:

```bash
gcloud auth login
gcloud config set project your-gcp-project

export REGION="us-central1"
export DATASET_IDS="legal-ko"
export OUTPUT_MODEL_REPO="your-hf-org/your-gemma4-e2b-fft"
export BURSTCHESTER_ACCESS_TOKEN="..."
export HF_TOKEN="..."

remote-trainer/submit-vertex-job.sh
```

The script enables required APIs, creates the Artifact Registry repository if needed, builds and pushes the Docker image, creates or updates Secret Manager token versions, prepares the Vertex service account, writes a CustomJob config, submits the job, waits for completion, and deletes the pushed Artifact Registry image after a successful training/upload run.

`BUILD_MODE=cloudbuild` is the default. Set `BUILD_MODE=docker` to build and push from the local Docker daemon.

Cleanup defaults:

```text
WAIT_FOR_COMPLETION=true
CLEANUP_ARTIFACT_IMAGE=true
CLEANUP_ARTIFACT_REPOSITORY=false
```

Set `CLEANUP_ARTIFACT_REPOSITORY=true` only when the repository is dedicated to this run.
