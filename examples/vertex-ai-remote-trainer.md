# Vertex AI Remote Trainer Example

This example runs the Burstchester remote trainer as a Vertex AI Custom Training job. The container downloads the base model from Hugging Face, downloads Burstchester datasets, trains with the settings you provide, uploads the trained output to Hugging Face, and can optionally register the model back to Burstchester.

Gemma 4 E2B full fine-tuning is memory-heavy. Start with `NVIDIA_A100_80GB` or larger. T4, L4, and A100 40GB are not expected to complete FFT reliably.

## 1. Configure gcloud

Run the Google Cloud login and project selection manually once:

```bash
gcloud auth login
gcloud config set project your-gcp-project
```

The submitting user must be allowed to enable services, create Artifact Registry repositories, use Cloud Build, manage Secret Manager secrets, create or use the Vertex service account, and submit Vertex AI custom jobs.

## 2. Set Training Inputs

Run from the `cli` directory:

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"

export DATASET_IDS="legal-ko"
export OUTPUT_MODEL_REPO="your-hf-org/your-gemma4-e2b-fft"

export BURSTCHESTER_ACCESS_TOKEN="..."
export HF_TOKEN="..."
```

`BURSTCHESTER_ACCESS_TOKEN` and `HF_TOKEN` are used only to create or update Secret Manager versions. They are not written into the Docker image or the generated job config.

If the secrets already exist and you do not want to add new versions, omit the token environment variables:

```bash
unset BURSTCHESTER_ACCESS_TOKEN
unset HF_TOKEN
```

The script will then use the existing latest versions of:

```text
burstchester-access-token
hf-token
```

## 3. Submit The Job

```bash
remote-trainer/submit-vertex-job.sh
```

The script performs the standardized execution sequence:

1. Enables `aiplatform`, `artifactregistry`, `cloudbuild`, and `secretmanager` APIs.
2. Creates the Artifact Registry Docker repository if needed.
3. Builds and pushes the trainer image.
4. Creates or updates Secret Manager token versions.
5. Creates the Vertex training service account if needed.
6. Grants the required project and secret access roles.
7. Generates a temporary Vertex CustomJob YAML.
8. Submits the Vertex AI custom job.
9. Waits for the job to finish.
10. Deletes the pushed Artifact Registry image after the job succeeds.

The trainer process exits only after the output model has been uploaded to Hugging Face, so a `JOB_STATE_SUCCEEDED` status means the upload completed. If the Vertex job fails or is cancelled, the script skips Artifact Registry cleanup so the image remains available for debugging.

`BUILD_MODE=cloudbuild` is the default and does not require local Docker. To build locally:

```bash
export BUILD_MODE="docker"
remote-trainer/submit-vertex-job.sh
```

## Common Settings

These values are optional and have defaults:

```bash
export REPOSITORY="burstchester-trainers"
export IMAGE_NAME="gemma4-remote-trainer"
export IMAGE_TAG="gemma4-full-trainer"
export JOB_DISPLAY_NAME="burstchester-gemma4-e2b-fft"

export BASE_MODEL="google/gemma-4-E2B"
export TRAIN_COMMAND="train-gemma4-e2b-full"
export TRAINING_METHOD="full"
export EPOCHS="1"
export BATCH_SIZE="1"
export MAX_SEQ_LENGTH="128"
export GRAD_ACCUM="8"
export LEARNING_RATE="0.00005"
export SKIP_REGISTER="true"
export MODEL_POINT_COST="100"
```

GPU defaults:

```bash
export MACHINE_TYPE="a2-ultragpu-1g"
export ACCELERATOR_TYPE="NVIDIA_A100_80GB"
export ACCELERATOR_COUNT="1"
export BOOT_DISK_TYPE="pd-ssd"
export BOOT_DISK_SIZE_GB="500"
```

Cleanup defaults:

```bash
export WAIT_FOR_COMPLETION="true"
export SKIP_IMAGE_BUILD="false"
export POLL_INTERVAL_SECONDS="60"
export CLEANUP_ARTIFACT_IMAGE="true"
export CLEANUP_ARTIFACT_REPOSITORY="false"
```

`CLEANUP_ARTIFACT_IMAGE=true` removes the pushed image tag after a successful run. Set `CLEANUP_ARTIFACT_REPOSITORY=true` only when the Artifact Registry repository is dedicated to this run, because repository deletion removes every image in that repository.

## Monitor Logs

```bash
gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT_ID}"
```

Use the job ID from the list:

```bash
gcloud ai custom-jobs describe JOB_ID --region="${REGION}" --project="${PROJECT_ID}"
```

Training logs are also available in Cloud Logging under Vertex AI Custom Jobs.

## Optional Burstchester Registration

To register the uploaded Hugging Face model back to Burstchester after training:

```bash
export SKIP_REGISTER="false"
export MODEL_POINT_COST="100"
remote-trainer/submit-vertex-job.sh
```
