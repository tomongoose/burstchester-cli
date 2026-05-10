# Vertex AI Remote Trainer Example

This example builds the Burstchester remote trainer Docker image, pushes it to Artifact Registry, and runs it as a Vertex AI Custom Training job. The container downloads the base model from Hugging Face, downloads Burstchester datasets, trains with the settings you provide, uploads the trained output to Hugging Face, and can optionally register the model back to Burstchester.

Gemma 4 E2B full fine-tuning is memory-heavy. Start with `NVIDIA_A100_80GB` or larger. T4, L4, and A100 40GB are not expected to complete FFT reliably.

## 1. Configure Shell Variables

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"
export REPOSITORY="burstchester-trainers"
export IMAGE_NAME="gemma4-remote-trainer"
export IMAGE_TAG="gemma4-full-trainer"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

export VERTEX_SERVICE_ACCOUNT="vertex-trainer@${PROJECT_ID}.iam.gserviceaccount.com"
export STAGING_BUCKET="gs://${PROJECT_ID}-vertex-staging"
```

## 2. Authenticate And Enable APIs

```bash
gcloud auth login
gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

## 3. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Burstchester trainer images"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

If the repository already exists, the create command can fail safely.

## 4. Build And Push The Image

Run from the `cli` directory:

```bash
docker build -f remote-trainer/Dockerfile -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
```

Cloud Build alternative:

```bash
gcloud builds submit . \
  --tag="${IMAGE_URI}" \
  --machine-type=e2-highcpu-32
```

## 5. Create Runtime Secrets

Do not put tokens in the Docker image.

```bash
printf "%s" "${BURSTCHESTER_ACCESS_TOKEN}" | \
  gcloud secrets create burstchester-access-token \
    --data-file=- \
    --replication-policy=automatic

printf "%s" "${HF_TOKEN}" | \
  gcloud secrets create hf-token \
    --data-file=- \
    --replication-policy=automatic
```

If the secrets already exist, add a new version instead:

```bash
printf "%s" "${BURSTCHESTER_ACCESS_TOKEN}" | \
  gcloud secrets versions add burstchester-access-token --data-file=-

printf "%s" "${HF_TOKEN}" | \
  gcloud secrets versions add hf-token --data-file=-
```

## 6. Create A Minimal Training Service Account

```bash
gcloud iam service-accounts create vertex-trainer \
  --display-name="Vertex Burstchester trainer"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding burstchester-access-token \
  --member="serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding hf-token \
  --member="serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

The caller that submits the job also needs permission to attach this service account, usually `roles/iam.serviceAccountUser` on the service account.

## 7. Create The Custom Job Config

Create `vertex-gemma4-job.yaml`:

```yaml
displayName: burstchester-gemma4-e2b-fft
jobSpec:
  serviceAccount: vertex-trainer@PROJECT_ID.iam.gserviceaccount.com
  workerPoolSpecs:
    - machineSpec:
        machineType: a2-ultragpu-1g
        acceleratorType: NVIDIA_A100_80GB
        acceleratorCount: 1
      replicaCount: 1
      diskSpec:
        bootDiskType: pd-ssd
        bootDiskSizeGb: 500
      containerSpec:
        imageUri: REGION-docker.pkg.dev/PROJECT_ID/burstchester-trainers/gemma4-remote-trainer:gemma4-full-trainer
        env:
          - name: DATASET_IDS
            value: legal-ko
          - name: OUTPUT_MODEL_REPO
            value: your-hf-org/your-gemma4-e2b-fft
          - name: BASE_MODEL
            value: google/gemma-4-E2B
          - name: TRAIN_COMMAND
            value: train-gemma4-e2b-full
          - name: EPOCHS
            value: "1"
          - name: BATCH_SIZE
            value: "1"
          - name: MAX_SEQ_LENGTH
            value: "128"
          - name: GRAD_ACCUM
            value: "8"
          - name: LEARNING_RATE
            value: "0.00005"
          - name: SKIP_REGISTER
            value: "true"
          - name: BURSTCHESTER_ACCESS_TOKEN_SECRET
            value: projects/PROJECT_ID/secrets/burstchester-access-token/versions/latest
          - name: HF_TOKEN_SECRET
            value: projects/PROJECT_ID/secrets/hf-token/versions/latest
```

Replace `PROJECT_ID`, `REGION`, and `OUTPUT_MODEL_REPO` values before submitting.

Vertex AI CustomJob `containerSpec` supports regular environment variables. The remote trainer reads the Secret Manager resource names above and fetches the token values at runtime using the attached service account.

## 8. Submit The Training Job

```bash
sed \
  -e "s/PROJECT_ID/${PROJECT_ID}/g" \
  -e "s/REGION/${REGION}/g" \
  vertex-gemma4-job.yaml > /tmp/vertex-gemma4-job.yaml

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --config=/tmp/vertex-gemma4-job.yaml
```

## 9. Monitor Logs

```bash
gcloud ai custom-jobs list --region="${REGION}"
```

Use the job ID from the list:

```bash
gcloud ai custom-jobs describe JOB_ID --region="${REGION}"
```

Training logs are also available in Cloud Logging under Vertex AI Custom Jobs.

## 10. Optional Burstchester Registration

Set these env values in the job config:

```yaml
- name: SKIP_REGISTER
  value: "false"
- name: MODEL_POINT_COST
  value: "100"
```

The container registers the uploaded Hugging Face model URL with Burstchester after a successful upload.
