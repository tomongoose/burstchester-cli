# Burstchester CLI

`cli/`는 Burstchester 데이터셋 다운로드와 로컬 파인튜닝 실행을 위한 경량 CLI다.

추가로 인증 부트스트랩도 포함한다. 현재 흐름은 다음 순서다.

1. CLI가 Firebase 익명 세션을 만든다.
2. 사용자가 `upsertCliProfile` 엔드포인트를 통해 Firestore 프로필을 만든다.

## 제공 명령

### 익명 로그인 후 Google 계정으로 업그레이드

기본값이 CLI에 내장되어 있다. 현재 `bustchester-e08c3` 프로젝트 기준으로 자동 설정되는 값은 다음이다.

- Firebase API key
- `upsertCliProfile`
- `prepareDatasetDownload`
- `debugUploadDataset`

다른 프로젝트를 쓰고 싶을 때만 대응 플래그나 env를 넘기면 된다.

```bash
node src/cli.mjs auth profile --display-name "Alice"
```

이 명령은:

- 로컬 세션이 없으면 Firebase 익명 로그인
- `upsertCliProfile` 엔드포인트로 Firestore 프로필 생성/병합
- 갱신된 세션을 `~/.burstchester/session.json`에 저장

세션 상태 확인:

```bash
node src/cli.mjs auth status
```

허깅페이스 토큰 저장:

```bash
node src/cli.mjs auth huggingface
```

또는 명시적으로 넘길 수 있다.

```bash
node src/cli.mjs auth huggingface --token hf_xxx
```

저장된 토큰 삭제:

```bash
node src/cli.mjs auth huggingface --clear
```

로그아웃:

```bash
node src/cli.mjs auth logout
```

### 데이터셋 리스트 관리

파인튜닝에 쓸 dataset ID 목록을 로컬 세션에 저장할 수 있다.

추가:

```bash
node src/cli.mjs dataset-list add --dataset-id legal-ko
node src/cli.mjs dataset-list add --dataset-id finance-ko
```

조회:

```bash
node src/cli.mjs dataset-list show
```

파일에서 import:

```bash
node src/cli.mjs dataset-list import --file ./dataset-ids.txt
```

파일로 export:

```bash
node src/cli.mjs dataset-list export --file ./dataset-ids.txt
```

삭제:

```bash
node src/cli.mjs dataset-list remove --dataset-id finance-ko
```

초기화:

```bash
node src/cli.mjs dataset-list clear
```

### 데이터셋 다운로드

```bash
node src/cli.mjs download-dataset \
  --dataset-id <dataset-id>
```

### Hugging Face 파일 다운로드

```bash
node src/cli.mjs download-model \
  --repo burstchester/legal-ko-qlora \
  --file adapter_model.safetensors
```

`download-model`은 토큰 우선순위를 다음 순서로 본다.

1. `--token`
2. 로컬에 저장된 Hugging Face 토큰
3. `HF_TOKEN`
4. `HUGGING_FACE_HUB_TOKEN`

또는 전체 URL을 직접 줄 수 있다.

```bash
node src/cli.mjs download-model \
  --url https://huggingface.co/burstchester/legal-ko-qlora/resolve/main/adapter_model.safetensors
```

### 디버그용 테스트 데이터 업로드

로컬 JSONL 파일을 백엔드의 디버그 업로드 엔드포인트로 보내서 실제 `datasets/{id}` 레코드를 생성한다.

```bash
node src/cli.mjs upload-test-dataset \
  --file ./fixtures/legal-ko.jsonl \
  --title "Legal Debug Dataset"
```

선택 플래그:

- `--dataset-id`
- `--title`
- `--description`
- `--tags`
- `--base-model-hint`
- `--task-type`
- `--language`
- `--license`
- `--source-model`
- `--output-model-id`
- `--upload-url`

기본 업로드 URL도 CLI에 내장되어 있으며 `debugUploadDataset` 함수를 가리킨다.

### 학습 실행

```bash
node src/cli.mjs train \
  --model-repo Qwen/Qwen3-0.6B
```

`train`은 우선순위를 다음 순서로 본다.

1. `--dataset-id`가 있으면 그 단일 dataset
2. 없으면 로컬에 저장된 dataset list 전체

저장된 dataset list를 쓰는 경우, 각 dataset ZIP을 백엔드에서 받아 `dataset.jsonl`을 추출하고, 이를 하나의 `merged-dataset.jsonl`로 합쳐서 Hugging Face 모델 학습에 넘긴다.

학습 전에 dataset list 전체를 검증만 하고 싶다면:

```bash
node src/cli.mjs train --model-repo Qwen/Qwen3-0.6B --preflight-only
```

이 모드는 각 dataset ID에 대해 백엔드 `prepareDatasetDownload` 호출이 성공하는지 먼저 확인하고, 성공/실패 목록을 JSON으로 출력한다.

### Gemma 4 E2B full fine-tuning

`google/gemma-4-E2B`를 고정해서 full fine-tuning 하는 전용 명령도 있다.

```bash
node src/cli.mjs train-gemma4-e2b-full --dataset-id legal-ko
```

기본 베이스 모델은 `google/gemma-4-E2B`지만, 필요하면 다른 Hugging Face repo를 넘길 수 있다.

```bash
node src/cli.mjs train-gemma4-e2b-full \
  --dataset-id legal-ko \
  --model-repo google/gemma-3-4b-it
```

또는 저장된 dataset list 전체를 그대로 사용할 수 있다.

```bash
node src/cli.mjs train-gemma4-e2b-full
```

이 명령은:

- dataset preflight 수행
- 각 dataset ZIP 다운로드
- `dataset.jsonl` 병합
- Python wrapper `src/python/train_gemma4_e2b_full.py` 실행
- 내부적으로 `trainingMethod=full`을 강제
- `--model-repo`가 없으면 `google/gemma-4-E2B`를 기본값으로 사용

### Gemma 2B IT LoRA fine-tuning

`llm/main.ipynb` 기준 설정을 반영한 `google/gemma-2b-it` 전용 LoRA 명령도 있다.

```bash
node src/cli.mjs train-gemma-2b-it-lora --dataset-id legal-ko
```

기본 베이스 모델은 `google/gemma-2b-it`지만, 필요하면 다른 Hugging Face repo를 넘길 수 있다.

```bash
node src/cli.mjs train-gemma-2b-it-lora \
  --dataset-id legal-ko \
  --model-repo google/gemma-3-1b-it
```

또는 저장된 dataset list 전체를 그대로 사용할 수 있다.

```bash
node src/cli.mjs train-gemma-2b-it-lora
```

이 명령은:

- dataset preflight 수행
- 각 dataset ZIP 다운로드
- `dataset.jsonl` 병합
- Python wrapper `src/python/train_gemma_2b_it_lora.py` 실행
- 내부적으로 `trainingMethod=lora`를 강제
- `--model-repo`가 없으면 `google/gemma-2b-it`를 기본값으로 사용
- 기본값으로 `maxSeqLength=128`, `loraRank=8`, `loraAlpha=16`, `loraDropout=0.05` 적용

## 학습 전제조건

CLI 자체는 Node 20 내장 기능만 사용한다. 실제 학습은 `python3`로 실행되며 아래 Python 패키지가 별도로 준비되어 있어야 한다.

- `torch`
- `transformers`
- `peft`
- `bitsandbytes` (`qlora` 사용 시)

`train` 명령은 백엔드에서 ZIP을 받아 `dataset.jsonl`을 추출한 뒤, `transformers`의 `from_pretrained` 경로를 통해 Hugging Face에서 `--model-repo` 모델을 내려받고 `src/python/train.py`로 로컬 파인튜닝을 실행한다.

### Google Colab 학습 + 모델 등록 스크립트

Colab에서는 저장소를 클론한 뒤 환경변수만 채워 스크립트를 실행할 수 있다.

```bash
git clone https://github.com/tomongoose/burstchester.git
cd burstchester

export BURSTCHESTER_FIREBASE_ID_TOKEN="<firebase-id-token>"
export HF_TOKEN="<huggingface-token>"
export DATASET_IDS="dataset-1,dataset-2"
export BASE_MODEL="google/gemma-2b-it"
export TRAIN_COMMAND="train-gemma-2b-it-lora"
export OUTPUT_MODEL_REPO="<hf-user-or-org>/<repo-name>"

bash cli/scripts/colab-train-and-register.sh
```

주요 옵션:

- `DATASET_IDS`: comma, space, newline 구분 dataset id 목록
- `BASE_MODEL`: 학습에 사용할 Hugging Face base model repo
- `TRAIN_COMMAND`: `train`, `train-gemma-2b-it-lora`, `train-gemma4-e2b-full`
- `OUTPUT_MODEL_REPO`: 학습 결과를 업로드할 Hugging Face model repo
- `OUTPUT_MODEL_URL`: 이미 업로드된 파일 URL을 등록할 때 사용
- `MODEL_POINT_COST`: 등록할 모델 다운로드 가격 포인트
- `SKIP_REGISTER=1`: 학습만 실행하고 백엔드 등록 생략
