#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required"
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    fail "set ${name}"
  fi
}

secret_resource() {
  local name="$1"
  printf 'projects/%s/secrets/%s/versions/latest' "${PROJECT_ID}" "${name}"
}

upsert_secret() {
  local secret_name="$1"
  local secret_value="$2"

  if [[ -z "${secret_value}" ]]; then
    printf 'Secret %s: using existing latest version\n' "${secret_name}"
    gcloud secrets describe "${secret_name}" --project="${PROJECT_ID}" >/dev/null
    return
  fi

  if gcloud secrets describe "${secret_name}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    printf 'Secret %s: adding a new version\n' "${secret_name}"
    printf '%s' "${secret_value}" | gcloud secrets versions add "${secret_name}" \
      --project="${PROJECT_ID}" \
      --data-file=-
  else
    printf 'Secret %s: creating\n' "${secret_name}"
    printf '%s' "${secret_value}" | gcloud secrets create "${secret_name}" \
      --project="${PROJECT_ID}" \
      --replication-policy=automatic \
      --data-file=-
  fi
}

add_project_role() {
  local role="$1"
  add_project_member_role "serviceAccount:${VERTEX_SERVICE_ACCOUNT}" "${role}"
}

add_project_member_role() {
  local member="$1"
  local role="$2"
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="${member}" \
    --role="${role}" \
    --condition=None \
    >/dev/null
}

add_secret_role() {
  local secret_name="$1"
  gcloud secrets add-iam-policy-binding "${secret_name}" \
    --project="${PROJECT_ID}" \
    --member="serviceAccount:${VERTEX_SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    >/dev/null
}

need_command gcloud

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
[[ -n "${PROJECT_ID}" && "${PROJECT_ID}" != "(unset)" ]] || fail "set PROJECT_ID or run gcloud config set project PROJECT_ID"
PROJECT_NUMBER="${PROJECT_NUMBER:-$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-$(gcloud config get-value account 2>/dev/null || true)}"

REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-burstchester-trainers}"
IMAGE_NAME="${IMAGE_NAME:-gemma4-remote-trainer}"
IMAGE_TAG="${IMAGE_TAG:-gemma4-full-trainer}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}}"
BUILD_MODE="${BUILD_MODE:-cloudbuild}"

VERTEX_SERVICE_ACCOUNT_NAME="${VERTEX_SERVICE_ACCOUNT_NAME:-vertex-trainer}"
VERTEX_SERVICE_ACCOUNT="${VERTEX_SERVICE_ACCOUNT:-${VERTEX_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"

BURSTCHESTER_ACCESS_TOKEN_SECRET_NAME="${BURSTCHESTER_ACCESS_TOKEN_SECRET_NAME:-burstchester-access-token}"
HF_TOKEN_SECRET_NAME="${HF_TOKEN_SECRET_NAME:-hf-token}"

MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-1g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_A100_80GB}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-500}"
JOB_DISPLAY_NAME="${JOB_DISPLAY_NAME:-burstchester-gemma4-e2b-fft}"

BASE_MODEL="${BASE_MODEL:-google/gemma-4-E2B}"
TRAIN_COMMAND="${TRAIN_COMMAND:-train-gemma4-e2b-full}"
TRAINING_METHOD="${TRAINING_METHOD:-full}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
SKIP_REGISTER="${SKIP_REGISTER:-true}"
MODEL_POINT_COST="${MODEL_POINT_COST:-100}"

require_env DATASET_IDS
require_env OUTPUT_MODEL_REPO

printf 'Project: %s\n' "${PROJECT_ID}"
printf 'Region: %s\n' "${REGION}"
printf 'Image: %s\n' "${IMAGE_URI}"
printf 'Vertex service account: %s\n' "${VERTEX_SERVICE_ACCOUNT}"

printf '\nEnabling required Google Cloud APIs...\n'
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT_ID}"

printf '\nEnsuring Artifact Registry repository exists...\n'
if ! gcloud artifacts repositories describe "${REPOSITORY}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --project="${PROJECT_ID}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Burstchester trainer images"
fi

printf '\nBuilding and pushing trainer image with %s...\n' "${BUILD_MODE}"
case "${BUILD_MODE}" in
  cloudbuild)
    gcloud builds submit "${CLI_DIR}" \
      --project="${PROJECT_ID}" \
      --tag="${IMAGE_URI}" \
      --machine-type=e2-highcpu-32
    ;;
  docker)
    need_command docker
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    docker build -f "${CLI_DIR}/remote-trainer/Dockerfile" -t "${IMAGE_URI}" "${CLI_DIR}"
    docker push "${IMAGE_URI}"
    ;;
  *)
    fail "BUILD_MODE must be cloudbuild or docker"
    ;;
esac

printf '\nEnsuring runtime secrets exist...\n'
upsert_secret "${BURSTCHESTER_ACCESS_TOKEN_SECRET_NAME}" "${BURSTCHESTER_ACCESS_TOKEN:-}"
upsert_secret "${HF_TOKEN_SECRET_NAME}" "${HF_TOKEN:-}"

printf '\nEnsuring Vertex service account and permissions...\n'
if ! gcloud iam service-accounts describe "${VERTEX_SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" \
  >/dev/null 2>&1; then
  gcloud iam service-accounts create "${VERTEX_SERVICE_ACCOUNT_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="Vertex Burstchester trainer"
fi

add_project_role "roles/aiplatform.user"
add_project_role "roles/artifactregistry.reader"
add_project_role "roles/storage.objectAdmin"
add_project_role "roles/secretmanager.secretAccessor"
add_secret_role "${BURSTCHESTER_ACCESS_TOKEN_SECRET_NAME}"
add_secret_role "${HF_TOKEN_SECRET_NAME}"

if [[ -n "${PROJECT_NUMBER}" ]]; then
  add_project_member_role "serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" "roles/artifactregistry.writer"
fi

if [[ -n "${GCLOUD_ACCOUNT}" && "${GCLOUD_ACCOUNT}" != "(unset)" ]]; then
  GCLOUD_ACCOUNT_MEMBER="user:${GCLOUD_ACCOUNT}"
  if [[ "${GCLOUD_ACCOUNT}" == *".gserviceaccount.com" ]]; then
    GCLOUD_ACCOUNT_MEMBER="serviceAccount:${GCLOUD_ACCOUNT}"
  fi
  gcloud iam service-accounts add-iam-policy-binding "${VERTEX_SERVICE_ACCOUNT}" \
    --project="${PROJECT_ID}" \
    --member="${GCLOUD_ACCOUNT_MEMBER}" \
    --role="roles/iam.serviceAccountUser" \
    >/dev/null || true
fi

JOB_CONFIG="$(mktemp "${TMPDIR:-/tmp}/burstchester-vertex-job.XXXXXX.yaml")"
cat >"${JOB_CONFIG}" <<EOF
displayName: ${JOB_DISPLAY_NAME}
jobSpec:
  serviceAccount: ${VERTEX_SERVICE_ACCOUNT}
  workerPoolSpecs:
    - machineSpec:
        machineType: ${MACHINE_TYPE}
        acceleratorType: ${ACCELERATOR_TYPE}
        acceleratorCount: ${ACCELERATOR_COUNT}
      replicaCount: 1
      diskSpec:
        bootDiskType: ${BOOT_DISK_TYPE}
        bootDiskSizeGb: ${BOOT_DISK_SIZE_GB}
      containerSpec:
        imageUri: ${IMAGE_URI}
        env:
          - name: DATASET_IDS
            value: "${DATASET_IDS}"
          - name: OUTPUT_MODEL_REPO
            value: "${OUTPUT_MODEL_REPO}"
          - name: BASE_MODEL
            value: "${BASE_MODEL}"
          - name: TRAIN_COMMAND
            value: "${TRAIN_COMMAND}"
          - name: TRAINING_METHOD
            value: "${TRAINING_METHOD}"
          - name: EPOCHS
            value: "${EPOCHS}"
          - name: BATCH_SIZE
            value: "${BATCH_SIZE}"
          - name: MAX_SEQ_LENGTH
            value: "${MAX_SEQ_LENGTH}"
          - name: GRAD_ACCUM
            value: "${GRAD_ACCUM}"
          - name: LEARNING_RATE
            value: "${LEARNING_RATE}"
          - name: SKIP_REGISTER
            value: "${SKIP_REGISTER}"
          - name: MODEL_POINT_COST
            value: "${MODEL_POINT_COST}"
          - name: BURSTCHESTER_ACCESS_TOKEN_SECRET
            value: "$(secret_resource "${BURSTCHESTER_ACCESS_TOKEN_SECRET_NAME}")"
          - name: HF_TOKEN_SECRET
            value: "$(secret_resource "${HF_TOKEN_SECRET_NAME}")"
EOF

if [[ -n "${DATASET_DOWNLOAD_URL:-}" ]]; then
  cat >>"${JOB_CONFIG}" <<EOF
          - name: DATASET_DOWNLOAD_URL
            value: "${DATASET_DOWNLOAD_URL}"
EOF
fi

if [[ -n "${MODEL_REGISTER_URL:-}" ]]; then
  cat >>"${JOB_CONFIG}" <<EOF
          - name: MODEL_REGISTER_URL
            value: "${MODEL_REGISTER_URL}"
EOF
fi

printf '\nSubmitting Vertex AI custom job...\n'
gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --config="${JOB_CONFIG}"

printf '\nSubmitted. Job config: %s\n' "${JOB_CONFIG}"
printf 'List jobs: gcloud ai custom-jobs list --project=%s --region=%s\n' "${PROJECT_ID}" "${REGION}"
