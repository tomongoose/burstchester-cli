import { resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { writeJson } from "./download.mjs";

const DEFAULT_TRAINING_METHOD = "qlora";

export function buildTrainingManifest(input) {
  return {
    datasetId: input.datasetId,
    datasetIds: Array.isArray(input.datasetIds) ? [...input.datasetIds] : [input.datasetId],
    datasetPath: input.datasetPath,
    modelRepo: input.modelRepo,
    outputDir: input.outputDir,
    trainingMethod: input.trainingMethod || DEFAULT_TRAINING_METHOD,
    numTrainEpochs: Number(input.numTrainEpochs ?? 1),
    perDeviceTrainBatchSize: Number(input.perDeviceTrainBatchSize ?? 1),
    gradientAccumulationSteps: Number(input.gradientAccumulationSteps ?? 4),
    learningRate: Number(input.learningRate ?? 0.0002),
    maxSeqLength: Number(input.maxSeqLength ?? 2048),
    loraRank: Number(input.loraRank ?? 16),
    loraAlpha: Number(input.loraAlpha ?? 32),
    loraDropout: Number(input.loraDropout ?? 0.05),
    loggingSteps: Number(input.loggingSteps ?? 10),
    saveSteps: Number(input.saveSteps ?? 100),
  };
}

export function buildGemma4E2BFullManifest(input) {
  return buildTrainingManifest({
    ...input,
    modelRepo: input.modelRepo ?? "google/gemma-4-E2B",
    trainingMethod: "full",
    learningRate: input.learningRate ?? 0.00005,
    gradientAccumulationSteps: input.gradientAccumulationSteps ?? 8,
    perDeviceTrainBatchSize: input.perDeviceTrainBatchSize ?? 1,
  });
}

export function buildGemma2BItLoraManifest(input) {
  return buildTrainingManifest({
    ...input,
    modelRepo: input.modelRepo ?? "google/gemma-2b-it",
    trainingMethod: "lora",
    numTrainEpochs: input.numTrainEpochs ?? 1,
    perDeviceTrainBatchSize: input.perDeviceTrainBatchSize ?? 1,
    gradientAccumulationSteps: input.gradientAccumulationSteps ?? 1,
    maxSeqLength: input.maxSeqLength ?? 128,
    loraRank: input.loraRank ?? 8,
    loraAlpha: input.loraAlpha ?? 16,
    loraDropout: input.loraDropout ?? 0.05,
  });
}

export function buildTrainingCommand({ pythonBin, scriptPath, configPath }) {
  return [pythonBin, scriptPath, "--config", configPath];
}

export async function runTraining({
  manifest,
  workDir,
  pythonBin = "python3",
  scriptPath = defaultTrainerScriptPath(),
  spawnImpl = spawn,
}) {
  const configPath = await writeJson(resolve(workDir, "train-config.json"), manifest);
  const command = buildTrainingCommand({
    pythonBin,
    scriptPath,
    configPath,
  });

  await new Promise((resolvePromise, rejectPromise) => {
    const child = spawnImpl(command[0], command.slice(1), {
      cwd: workDir,
      stdio: "inherit",
      env: process.env,
    });

    child.on("error", rejectPromise);
    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise();
        return;
      }

      rejectPromise(new Error(`Training process exited with code ${code}.`));
    });
  });

  return {
    configPath,
    command,
  };
}

export function defaultTrainerScriptPath() {
  return fileURLToPath(new URL("../python/train.py", import.meta.url));
}

export function defaultGemma4FullTrainerScriptPath() {
  return fileURLToPath(new URL("../python/train_gemma4_e2b_full.py", import.meta.url));
}

export function defaultGemma2BItLoraTrainerScriptPath() {
  return fileURLToPath(new URL("../python/train_gemma_2b_it_lora.py", import.meta.url));
}
