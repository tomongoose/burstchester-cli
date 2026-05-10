import test from "node:test";
import assert from "node:assert/strict";

import {
  buildDatasetDownloadUrl,
  buildIssueAccessTokenRequest,
  buildRegisterModelRequest,
  buildRecordModelDownloadRequest,
  buildUpdateAssetPointCostRequest,
} from "../src/lib/backend.mjs";

test("buildDatasetDownloadUrl appends datasetId query parameter", () => {
  const url = buildDatasetDownloadUrl(
    "https://us-central1-demo.cloudfunctions.net/prepareDatasetDownload",
    "dataset-1",
  );

  assert.equal(
    url,
    "https://us-central1-demo.cloudfunctions.net/prepareDatasetDownload?datasetId=dataset-1",
  );
});

test("buildIssueAccessTokenRequest posts an authenticated token label", () => {
  const request = buildIssueAccessTokenRequest({
    endpointUrl: "https://functions.example/issueAccessToken",
    idToken: "firebase-id-token",
    label: "Colab",
  });

  assert.equal(request.url, "https://functions.example/issueAccessToken");
  assert.equal(request.options.headers.authorization, "Bearer firebase-id-token");
  assert.deepEqual(JSON.parse(request.options.body), { label: "Colab" });
});

test("buildRecordModelDownloadRequest posts authenticated model purchase metadata", () => {
  const request = buildRecordModelDownloadRequest({
    endpointUrl: "https://functions.example/recordModelDownload",
    idToken: "firebase-id-token",
    modelName: "Qwen/Qwen3-0.6B",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.gguf",
  });

  assert.equal(request.url, "https://functions.example/recordModelDownload");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers.authorization, "Bearer firebase-id-token");
  assert.deepEqual(JSON.parse(request.options.body), {
    modelName: "Qwen/Qwen3-0.6B",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.gguf",
  });
});

test("buildRegisterModelRequest posts paid training metadata", () => {
  const request = buildRegisterModelRequest({
    endpointUrl: "https://functions.example/registerModelHttp",
    idToken: "firebase-id-token",
    huggingFaceUrl: "https://huggingface.co/user/model/resolve/main/model.gguf",
    baseModel: "Qwen/Qwen3-0.6B",
    trainingDatasets: ["dataset-1"],
    trainingMethod: "qlora",
    pointCost: 250,
    ollamaPullUrl: "burstchester/model:latest",
  });

  assert.equal(request.url, "https://functions.example/registerModelHttp");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(JSON.parse(request.options.body), {
    huggingFaceUrl: "https://huggingface.co/user/model/resolve/main/model.gguf",
    baseModel: "Qwen/Qwen3-0.6B",
    trainingDatasets: ["dataset-1"],
    trainingMethod: "qlora",
    pointCost: 250,
    ollamaPullUrl: "burstchester/model:latest",
  });
});

test("buildUpdateAssetPointCostRequest posts owner price changes", () => {
  const request = buildUpdateAssetPointCostRequest({
    endpointUrl: "https://functions.example/updateAssetPointCost",
    idToken: "firebase-id-token",
    assetType: "dataset",
    assetId: "dataset-1",
    pointCost: 25,
  });

  assert.equal(request.url, "https://functions.example/updateAssetPointCost");
  assert.equal(request.options.headers.authorization, "Bearer firebase-id-token");
  assert.deepEqual(JSON.parse(request.options.body), {
    assetType: "dataset",
    assetId: "dataset-1",
    pointCost: 25,
  });
});
