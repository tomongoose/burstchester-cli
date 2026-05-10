#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  BURSTCHESTER_ACCESS_TOKEN=<web_cli_access_token> \
  HF_TOKEN=<huggingface_token> \
  DATASET_IDS="dataset-1,dataset-2" \
  BASE_MODEL="Qwen/Qwen3-0.6B" \
  ./scripts/colab-train-and-register.sh

Required env:
  BURSTCHESTER_ACCESS_TOKEN       Burstchester web-issued CLI access token.
  DATASET_IDS                     Comma, space, or newline separated dataset IDs.
  BASE_MODEL                      Hugging Face model repo used as the training base model.

Optional env:
  BURSTCHESTER_FIREBASE_ID_TOKEN  Firebase ID token for legacy session-based auth.
  HF_TOKEN                        Hugging Face token for gated models and upload.
  TRAIN_COMMAND                   train | train-gemma-2b-it-lora | train-gemma4-e2b-full.
  TRAINING_METHOD                 qlora | lora | full, used with TRAIN_COMMAND=train.
  WORKSPACE                       Training workspace directory.
  PYTHON_BIN                      Python executable.
  EPOCHS                          Forwarded to CLI --epochs.
  BATCH_SIZE                      Forwarded to CLI --batch-size.
  MAX_SEQ_LENGTH                  Forwarded to CLI --max-seq-length.
  LORA_RANK                       Forwarded to CLI --lora-rank.
  LORA_ALPHA                      Forwarded to CLI --lora-alpha.
  LORA_DROPOUT                    Forwarded to CLI --lora-dropout.
  OUTPUT_MODEL_REPO               Hugging Face repo to upload the trained output to.
  OUTPUT_MODEL_URL                Existing Hugging Face downloadable URL to register.
  MODEL_POINT_COST                Point cost for the registered output model.
  OLLAMA_PULL_URL                 Optional Ollama pull URL stored with the model record.
  SKIP_REGISTER=1                 Train only, skip backend model registration.
  SKIP_HF_UPLOAD=1                Do not upload output to Hugging Face.
EOF
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env: ${name}" >&2
    usage >&2
    exit 2
  fi
}

write_session() {
  local session_path="${HOME}/.burstchester/session.json"
  mkdir -p "$(dirname "${session_path}")"
  node --input-type=module - "${session_path}" "${BURSTCHESTER_FIREBASE_ID_TOKEN}" <<'NODE'
import { writeFile } from "node:fs/promises";

const [sessionPath, idToken] = process.argv.slice(2);
const [, payloadPart] = idToken.split(".");
if (!payloadPart) {
  throw new Error("BURSTCHESTER_FIREBASE_ID_TOKEN is not a JWT.");
}
const payload = JSON.parse(Buffer.from(payloadPart, "base64url").toString("utf8"));
const expiresAt = Number(payload.exp || 0) > 0
  ? Number(payload.exp) * 1000
  : Date.now() + 55 * 60 * 1000;

await writeFile(
  sessionPath,
  `${JSON.stringify({
    userId: payload.user_id || payload.sub,
    idToken,
    refreshToken: "",
    expiresAt,
    isAnonymous: payload.firebase?.sign_in_provider === "anonymous",
    providerId: payload.firebase?.sign_in_provider || "unknown",
    email: payload.email || "",
  }, null, 2)}\n`,
  "utf8",
);
NODE
}

normalize_dataset_list() {
  node --input-type=module - "${DATASET_IDS}" <<'NODE'
const raw = process.argv[2] || "";
const ids = Array.from(
  new Set(raw.split(/[\s,]+/).map((value) => value.trim()).filter(Boolean)),
);
if (ids.length === 0) {
  throw new Error("DATASET_IDS did not contain any dataset IDs.");
}
process.stdout.write(`${ids.join("\n")}\n`);
NODE
}

install_python_deps() {
  # Colab ships a CUDA-matched torch/torchvision/torchaudio stack. Upgrading
  # torch here can pull a different CUDA toolkit and break the runtime.
  "${PYTHON_BIN}" -m pip install -q -U \
    transformers \
    peft \
    accelerate \
    datasets \
    huggingface_hub
  "${PYTHON_BIN}" -m pip install -q -U --no-deps bitsandbytes
}

upload_output_model() {
  local output_dir="$1"
  local repo="$2"
  "${PYTHON_BIN}" - "${output_dir}" "${repo}" <<'PY'
import os
import sys
from huggingface_hub import HfApi, create_repo

output_dir, repo = sys.argv[1:3]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for Hugging Face upload.")

create_repo(repo_id=repo, token=token, exist_ok=True)
api = HfApi(token=token)
api.upload_folder(folder_path=output_dir, repo_id=repo, repo_type="model")
print(f"https://huggingface.co/{repo}")
PY
}

main() {
  if [[ -z "${BURSTCHESTER_ACCESS_TOKEN:-}" && -z "${BURSTCHESTER_FIREBASE_ID_TOKEN:-}" ]]; then
    echo "Missing required env: BURSTCHESTER_ACCESS_TOKEN or BURSTCHESTER_FIREBASE_ID_TOKEN" >&2
    usage >&2
    exit 2
  fi
  require_env DATASET_IDS
  require_env BASE_MODEL

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${script_dir}/../src/cli.mjs" ]]; then
    cd "${script_dir}/.."
  elif [[ -f "${script_dir}/../../cli/src/cli.mjs" ]]; then
    cd "${script_dir}/../../cli"
  else
    echo "Could not locate Burstchester CLI root from ${script_dir}" >&2
    exit 2
  fi

  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  TRAIN_COMMAND="${TRAIN_COMMAND:-train-gemma-2b-it-lora}"
  WORKSPACE="${WORKSPACE:-/content/burstchester-training}"

  if [[ -n "${BURSTCHESTER_FIREBASE_ID_TOKEN:-}" ]]; then
    write_session
  fi
  normalize_dataset_list > "${WORKSPACE}.dataset-ids.txt"
  node src/cli.mjs dataset-list import --file "${WORKSPACE}.dataset-ids.txt" >/dev/null

  install_python_deps

  local train_args=(
    "${TRAIN_COMMAND}"
    "--model-repo" "${BASE_MODEL}"
    "--workspace" "${WORKSPACE}"
    "--python" "${PYTHON_BIN}"
  )

  [[ -n "${TRAINING_METHOD:-}" ]] && train_args+=("--training-method" "${TRAINING_METHOD}")
  [[ -n "${EPOCHS:-}" ]] && train_args+=("--epochs" "${EPOCHS}")
  [[ -n "${BATCH_SIZE:-}" ]] && train_args+=("--batch-size" "${BATCH_SIZE}")
  [[ -n "${MAX_SEQ_LENGTH:-}" ]] && train_args+=("--max-seq-length" "${MAX_SEQ_LENGTH}")
  [[ -n "${LORA_RANK:-}" ]] && train_args+=("--lora-rank" "${LORA_RANK}")
  [[ -n "${LORA_ALPHA:-}" ]] && train_args+=("--lora-alpha" "${LORA_ALPHA}")
  [[ -n "${LORA_DROPOUT:-}" ]] && train_args+=("--lora-dropout" "${LORA_DROPOUT}")

  node src/cli.mjs "${train_args[@]}" | tee "${WORKSPACE}.train-result.json"

  local output_dir
  output_dir="$(node --input-type=module - "${WORKSPACE}.train-result.json" <<'NODE'
import { readFileSync } from "node:fs";
const text = readFileSync(process.argv[2], "utf8");
const match = text.match(/\{[\s\S]*\}\s*$/);
if (!match) throw new Error("Could not parse training result JSON.");
const payload = JSON.parse(match[0]);
process.stdout.write(payload.outputDir || `${process.argv[2]}.output`);
NODE
)"

  if [[ "${SKIP_REGISTER:-0}" == "1" ]]; then
    exit 0
  fi

  local model_url="${OUTPUT_MODEL_URL:-}"
  if [[ -z "${model_url}" && "${SKIP_HF_UPLOAD:-0}" != "1" ]]; then
    require_env OUTPUT_MODEL_REPO
    upload_output_model "${output_dir}" "${OUTPUT_MODEL_REPO}" | tee "${WORKSPACE}.hf-upload.txt"
    model_url="https://huggingface.co/${OUTPUT_MODEL_REPO}/resolve/main/adapter_model.safetensors"
  fi
  if [[ -z "${model_url}" ]]; then
    echo "Set OUTPUT_MODEL_URL or OUTPUT_MODEL_REPO, or set SKIP_REGISTER=1." >&2
    exit 2
  fi

  node src/cli.mjs register-model \
    --huggingface-url "${model_url}" \
    --base-model "${BASE_MODEL}" \
    --dataset-file "${WORKSPACE}.dataset-ids.txt" \
    ${MODEL_POINT_COST:+--point-cost "${MODEL_POINT_COST}"} \
    ${OLLAMA_PULL_URL:+--ollama-pull-url "${OLLAMA_PULL_URL}"} \
    | tee "${WORKSPACE}.register-result.json"
}

main "$@"
