import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";

import {
  buildGemma2BItLoraManifest,
  buildGemma4E2BFullManifest,
  buildTrainingCommand,
  buildTrainingManifest,
  defaultGemma2BItLoraTrainerScriptPath,
} from "../src/lib/train.mjs";

test("buildTrainingManifest keeps core training metadata", () => {
  const manifest = buildTrainingManifest({
    datasetId: "dataset-1",
    datasetIds: ["dataset-1", "dataset-2"],
    datasetPath: "/tmp/dataset.jsonl",
    modelRepo: "Qwen/Qwen3-0.6B",
    outputDir: "/tmp/out",
    trainingMethod: "qlora",
  });

  assert.equal(manifest.datasetId, "dataset-1");
  assert.deepEqual(manifest.datasetIds, ["dataset-1", "dataset-2"]);
  assert.equal(manifest.datasetPath, "/tmp/dataset.jsonl");
  assert.equal(manifest.modelRepo, "Qwen/Qwen3-0.6B");
  assert.equal(manifest.outputDir, "/tmp/out");
  assert.equal(manifest.trainingMethod, "qlora");
});

test("buildTrainingCommand points at the bundled python trainer", () => {
  const command = buildTrainingCommand({
    pythonBin: "python3",
    scriptPath: "/workspace/cli/src/python/train.py",
    configPath: "/tmp/train-config.json",
  });

  assert.deepEqual(command, [
    "python3",
    "/workspace/cli/src/python/train.py",
    "--config",
    "/tmp/train-config.json",
  ]);
});

test("buildGemma4E2BFullManifest pins model repo and full training mode", () => {
  const manifest = buildGemma4E2BFullManifest({
    datasetId: "dataset-1",
    datasetIds: ["dataset-1"],
    datasetPath: "/tmp/merged-dataset.jsonl",
    outputDir: "/tmp/out",
  });

  assert.equal(manifest.modelRepo, "google/gemma-4-E2B");
  assert.equal(manifest.trainingMethod, "full");
  assert.deepEqual(manifest.datasetIds, ["dataset-1"]);
});

test("buildGemma4E2BFullManifest allows an explicit base model override", () => {
  const manifest = buildGemma4E2BFullManifest({
    datasetId: "dataset-1",
    datasetIds: ["dataset-1"],
    datasetPath: "/tmp/merged-dataset.jsonl",
    outputDir: "/tmp/out",
    modelRepo: "google/gemma-3-4b-it",
  });

  assert.equal(manifest.modelRepo, "google/gemma-3-4b-it");
  assert.equal(manifest.trainingMethod, "full");
});

test("buildGemma2BItLoraManifest pins model repo and notebook-style lora defaults", () => {
  const manifest = buildGemma2BItLoraManifest({
    datasetId: "dataset-1",
    datasetIds: ["dataset-1"],
    datasetPath: "/tmp/merged-dataset.jsonl",
    outputDir: "/tmp/out",
  });

  assert.equal(manifest.modelRepo, "google/gemma-2b-it");
  assert.equal(manifest.trainingMethod, "lora");
  assert.equal(manifest.maxSeqLength, 128);
  assert.equal(manifest.loraRank, 8);
  assert.equal(manifest.loraAlpha, 16);
  assert.equal(manifest.loraDropout, 0.05);
  assert.equal(manifest.perDeviceTrainBatchSize, 1);
  assert.equal(manifest.gradientAccumulationSteps, 1);
  assert.equal(manifest.numTrainEpochs, 1);
  assert.deepEqual(manifest.datasetIds, ["dataset-1"]);
});

test("buildGemma2BItLoraManifest allows an explicit base model override", () => {
  const manifest = buildGemma2BItLoraManifest({
    datasetId: "dataset-1",
    datasetIds: ["dataset-1"],
    datasetPath: "/tmp/merged-dataset.jsonl",
    outputDir: "/tmp/out",
    modelRepo: "google/gemma-3-1b-it",
  });

  assert.equal(manifest.modelRepo, "google/gemma-3-1b-it");
  assert.equal(manifest.trainingMethod, "lora");
  assert.equal(manifest.maxSeqLength, 128);
});

test("defaultGemma2BItLoraTrainerScriptPath points at the dedicated wrapper", () => {
  const scriptPath = defaultGemma2BItLoraTrainerScriptPath();
  assert.match(
    scriptPath,
    /train_gemma_2b_it_lora\.py$/,
  );
  assert.equal(existsSync(scriptPath), true);
});
