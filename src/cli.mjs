#!/usr/bin/env node

import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline/promises";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { parseArgs, requiredFlag } from "./lib/args.mjs";
import {
  fetchDatasetPackageMetadata,
  issueAccessToken,
  preflightDatasetDownloads,
  registerModel,
  recordModelDownload,
  updateAssetPointCost,
  uploadDebugDataset,
} from "./lib/backend.mjs";
import { BURSTCHESTER_DEFAULTS } from "./lib/default-config.mjs";
import { parseDatasetIdFile, serializeDatasetIds } from "./lib/dataset-list.mjs";
import { resolveDatasetIdsInput } from "./lib/dataset-list-source.mjs";
import { downloadToFile, ensureDir, mergeTextFiles } from "./lib/download.mjs";
import {
  isSessionExpired,
  refreshFirebaseSession,
  signInAnonymously,
} from "./lib/firebase-auth.mjs";
import { buildHuggingFaceFileUrl, downloadHuggingFaceFile } from "./lib/huggingface.mjs";
import { upsertCliProfile } from "./lib/profile.mjs";
import {
  appendProxyJsonlSample,
  buildProxyLogSample,
  buildProxyUploadMetadata,
  isStreamingProxyRequest,
  normalizeProxyRequestBody,
} from "./lib/proxy-log.mjs";
import {
  addDatasetId,
  clearDatasetIds,
  clearSession,
  loadAccessToken,
  loadSession,
  normalizeDatasetId,
  removeDatasetId,
  saveSession,
} from "./lib/session.mjs";
import {
  buildGemma2BItLoraManifest,
  buildGemma4E2BFullManifest,
  buildTrainingManifest,
  defaultGemma2BItLoraTrainerScriptPath,
  defaultGemma4FullTrainerScriptPath,
  runTraining,
} from "./lib/train.mjs";
import { extractStoredZip } from "./lib/zip.mjs";

const ROOT_DIR = resolve(fileURLToPath(new URL("..", import.meta.url)));

async function main(argv) {
  const { command, flags, positionals } = parseArgs(argv);

  switch (command) {
    case "auth":
      await handleAuth(flags, positionals);
      return;
    case "access-token":
      await handleAccessToken(flags, positionals);
      return;
    case "download-dataset":
      await handleDownloadDataset(flags);
      return;
    case "download-model":
      await handleDownloadModel(flags);
      return;
    case "dataset-list":
      await handleDatasetList(flags, positionals);
      return;
    case "upload-test-dataset":
      await handleUploadTestDataset(flags);
      return;
    case "proxy-record":
      await handleProxyRecord(flags);
      return;
    case "upload-proxy-log":
      await handleUploadProxyLog(flags);
      return;
    case "register-model":
      await handleRegisterModel(flags);
      return;
    case "update-point-cost":
      await handleUpdatePointCost(flags);
      return;
    case "train":
      await handleTrain(flags);
      return;
    case "train-gemma4-e2b-full":
      await handleTrainGemma4E2BFull(flags);
      return;
    case "train-gemma-2b-it-lora":
      await handleTrainGemma2BItLora(flags);
      return;
    default:
      printUsage();
  }
}

async function handleDatasetList(flags, positionals) {
  const subcommand = positionals[0] || "show";
  const session = (await loadSession()) ?? {};
  const currentIds = Array.isArray(session.datasetIds) ? session.datasetIds : [];

  switch (subcommand) {
    case "add": {
      const datasetId = requiredFlag(flags, "dataset-id");
      session.datasetIds = addDatasetId(currentIds, datasetId);
      await saveSession(session);
      process.stdout.write(`${JSON.stringify({ ok: true, datasetIds: session.datasetIds }, null, 2)}\n`);
      return;
    }
    case "remove": {
      const datasetId = requiredFlag(flags, "dataset-id");
      session.datasetIds = removeDatasetId(currentIds, datasetId);
      await saveSession(session);
      process.stdout.write(`${JSON.stringify({ ok: true, datasetIds: session.datasetIds }, null, 2)}\n`);
      return;
    }
    case "clear": {
      session.datasetIds = clearDatasetIds(currentIds);
      await saveSession(session);
      process.stdout.write(`${JSON.stringify({ ok: true, datasetIds: [] }, null, 2)}\n`);
      return;
    }
    case "import": {
      const filePath = requiredFlag(flags, "file");
      const text = await readFile(resolve(filePath), "utf8");
      session.datasetIds = parseDatasetIdFile(text);
      await saveSession(session);
      process.stdout.write(`${JSON.stringify({ ok: true, imported: true, datasetIds: session.datasetIds }, null, 2)}\n`);
      return;
    }
    case "export": {
      const filePath = requiredFlag(flags, "file");
      const text = serializeDatasetIds(currentIds);
      await writeFile(resolve(filePath), text, "utf8");
      process.stdout.write(`${JSON.stringify({ ok: true, exported: true, file: resolve(filePath), datasetIds: currentIds }, null, 2)}\n`);
      return;
    }
    case "show": {
      process.stdout.write(`${JSON.stringify({ ok: true, datasetIds: currentIds }, null, 2)}\n`);
      return;
    }
    default:
      throw new Error(`Unknown dataset-list subcommand: ${subcommand}`);
  }
}

async function handleAuth(flags, positionals) {
  const subcommand = positionals[0] || "status";

  switch (subcommand) {
    case "status":
      await handleAuthStatus(flags);
      return;
    case "huggingface":
      await handleAuthHuggingFace(flags);
      return;
    case "profile":
      await handleAuthProfile(flags);
      return;
    case "logout":
      await clearSession();
      process.stdout.write(`${JSON.stringify({ ok: true, signedOut: true }, null, 2)}\n`);
      return;
    default:
      throw new Error(`Unknown auth subcommand: ${subcommand}`);
  }
}

async function handleAccessToken(flags, positionals) {
  const subcommand = positionals[0] || "issue";

  switch (subcommand) {
    case "issue": {
      const session = await loadActiveSessionForBackendWrite(flags);
      const endpointUrl = resolveConfig(
        flags["token-url"],
        process.env.BURSTCHESTER_ACCESS_TOKEN_URL,
        BURSTCHESTER_DEFAULTS.accessTokenUrl,
      );
      const label = typeof flags.label === "string" && flags.label.trim()
        ? flags.label.trim()
        : "CLI access token";
      const issued = await issueAccessToken({
        endpointUrl,
        idToken: session.idToken,
        label,
      });
      process.stdout.write(`${JSON.stringify(issued, null, 2)}\n`);
      return;
    }
    default:
      throw new Error(`Unknown access-token subcommand: ${subcommand}`);
  }
}

async function handleAuthStatus(flags) {
  let session = await loadSession();
  const apiKey = resolveConfig(
    flags["api-key"],
    process.env.BURSTCHESTER_FIREBASE_API_KEY,
    BURSTCHESTER_DEFAULTS.firebaseConfig.apiKey,
  );

  if (!session) {
    process.stdout.write(`${JSON.stringify({ signedIn: false }, null, 2)}\n`);
    return;
  }

  if (apiKey && isSessionExpired(session)) {
    session = {
      ...session,
      ...await refreshFirebaseSession({
        apiKey,
        refreshToken: session.refreshToken,
      }),
    };
    await saveSession(session);
  }

  process.stdout.write(
    `${JSON.stringify(
      {
        signedIn: true,
        userId: session.userId,
        isAnonymous: Boolean(session.isAnonymous),
        providerId: session.providerId,
        email: session.email || "",
      },
      null,
      2,
    )}\n`,
  );
}

async function handleAuthHuggingFace(flags) {
  const session = (await loadSession()) ?? {};

  if (flags.clear === true) {
    delete session.huggingFaceToken;
    await saveSession(session);
    process.stdout.write(`${JSON.stringify({ ok: true, cleared: true }, null, 2)}\n`);
    return;
  }

  let token = typeof flags.token === "string" ? flags.token : "";
  if (!token.trim()) {
    token = await promptForToken("Hugging Face token: ");
  }

  const normalized = normalizeLocalToken(token);
  if (!normalized) {
    throw new Error("Hugging Face token must not be empty.");
  }

  session.huggingFaceToken = normalized;
  await saveSession(session);
  process.stdout.write(
    `${JSON.stringify({ ok: true, stored: true, tokenPreview: `${normalized.slice(0, 6)}...` }, null, 2)}\n`,
  );
}

async function handleAuthProfile(flags) {
  const apiKey = requiredConfig(
    flags["api-key"],
    process.env.BURSTCHESTER_FIREBASE_API_KEY,
    "api-key",
    BURSTCHESTER_DEFAULTS.firebaseConfig.apiKey,
  );
  const displayName = requiredFlag(flags, "display-name");
  const photoURL = optionalConfig(flags["photo-url"], process.env.BURSTCHESTER_PROFILE_PHOTO_URL);
  const profileUrl = resolveConfig(
    flags["profile-url"],
    process.env.BURSTCHESTER_PROFILE_URL,
    BURSTCHESTER_DEFAULTS.profileUrl,
  );

  let session = await loadSession();
  if (!session) {
    session = await signInAnonymously({ apiKey });
  } else if (isSessionExpired(session)) {
    session = {
      ...session,
      ...await refreshFirebaseSession({
        apiKey,
        refreshToken: session.refreshToken,
      }),
    };
  }

  const profile = await upsertCliProfile({
    profileUrl,
    idToken: session.idToken,
    displayName,
    photoURL,
  });

  session.displayName = profile.displayName;
  session.photoURL = profile.photoURL || null;
  session.email = profile.email || session.email || "";
  await saveSession(session);

  process.stdout.write(
    `${JSON.stringify(
      {
        ok: true,
        upgraded: false,
        auth: {
          userId: session.userId,
          isAnonymous: session.isAnonymous,
          providerId: session.providerId,
          email: session.email || "",
        },
        profile,
      },
      null,
      2,
    )}\n`,
  );
}

async function loadBackendAuthForDownload(flags) {
  const accessToken = resolveConfig(
    flags["access-token"],
    process.env.BURSTCHESTER_ACCESS_TOKEN,
    await loadAccessToken(),
  );

  if (accessToken) {
    return {
      bearerToken: accessToken,
      session: await loadSession(),
    };
  }

  const session = await loadActiveSessionForBackendWrite(flags);
  return {
    bearerToken: session.idToken,
    session,
  };
}

async function handleDownloadDataset(flags) {
  const endpointUrl = resolveConfig(
    flags["backend-url"],
    process.env.BURSTCHESTER_BACKEND_URL,
    BURSTCHESTER_DEFAULTS.datasetDownloadUrl,
  );
  const datasetId = requiredFlag(flags, "dataset-id");
  const auth = await loadBackendAuthForDownload(flags);
  const outDir = resolve(String(flags["out-dir"] || join(ROOT_DIR, "artifacts", "datasets", datasetId)));
  const extract = flags.extract !== "false";

  const metadata = await fetchDatasetPackageMetadata({
    endpointUrl,
    datasetId,
    idToken: auth.bearerToken,
  });

  await ensureDir(outDir);
  const zipPath = join(outDir, `${datasetId}.zip`);
  await downloadToFile({
    url: metadata.url,
    destination: zipPath,
  });

  let extractedDir = null;
  if (extract) {
    extractedDir = join(outDir, datasetId);
    const archive = await readFile(zipPath);
    await extractStoredZip(archive, extractedDir);
  }

  process.stdout.write(
    `${JSON.stringify({ datasetId, zipPath, extractedDir, downloadUrl: metadata.url }, null, 2)}\n`,
  );
}

async function handleDownloadModel(flags) {
  const outDir = resolve(String(flags["out-dir"] || join(ROOT_DIR, "artifacts", "models")));
  const url = typeof flags.url === "string" ? flags.url : undefined;
  const repo = typeof flags.repo === "string" ? flags.repo : undefined;
  const file = typeof flags.file === "string" ? flags.file : undefined;
  const revision = typeof flags.revision === "string" ? flags.revision : "main";
  const auth = await loadBackendAuthForDownload(flags);
  const recordUrl = resolveConfig(
    flags["record-url"],
    process.env.BURSTCHESTER_MODEL_DOWNLOAD_URL,
    BURSTCHESTER_DEFAULTS.modelDownloadUrl,
  );

  if (!url && !(repo && file)) {
    throw new Error("download-model requires --url or --repo with --file");
  }

  const sourceUrl = url || buildHuggingFaceFileUrl(repo, file, revision);
  const modelName = typeof flags["model-name"] === "string" && flags["model-name"].trim()
    ? flags["model-name"].trim()
    : repo || sourceUrl;
  const purchase = await recordModelDownload({
    endpointUrl: recordUrl,
    idToken: auth.bearerToken,
    modelName,
    sourceUrl,
  });
  const result = await downloadHuggingFaceFile({
    url,
    repo,
    file,
    revision,
    outDir,
    token: resolveDownloadToken(flags, auth.session),
  });

  process.stdout.write(`${JSON.stringify({ ...result, modelName, purchase }, null, 2)}\n`);
}

async function handleUploadTestDataset(flags) {
  const auth = await loadBackendAuthForDownload(flags);

  const endpointUrl = resolveConfig(
    flags["upload-url"],
    process.env.BURSTCHESTER_DEBUG_UPLOAD_URL,
    BURSTCHESTER_DEFAULTS.debugUploadUrl,
  );
  const filePath = requiredFlag(flags, "file");
  const filename = typeof flags.filename === "string" && flags.filename.trim()
    ? flags.filename.trim()
    : filePath.split("/").at(-1) || "debug.jsonl";
  const content = await readFile(resolve(filePath), "utf8");

  const dataset = await uploadDebugDataset({
    endpointUrl,
    idToken: auth.bearerToken,
    filename,
    content,
    metadata: {
      title: typeof flags.title === "string" ? flags.title : undefined,
      description: typeof flags.description === "string" ? flags.description : undefined,
      tags: typeof flags.tags === "string" ? flags.tags : undefined,
      baseModelHint: typeof flags["base-model-hint"] === "string" ? flags["base-model-hint"] : undefined,
      taskType: typeof flags["task-type"] === "string" ? flags["task-type"] : undefined,
      language: typeof flags.language === "string" ? flags.language : undefined,
      license: typeof flags.license === "string" ? flags.license : undefined,
      sourceModel: typeof flags["source-model"] === "string" ? flags["source-model"] : undefined,
      outputModelId: typeof flags["output-model-id"] === "string" ? flags["output-model-id"] : undefined,
      pointCost: typeof flags["point-cost"] === "string" ? flags["point-cost"] : undefined,
    },
  });

  process.stdout.write(`${JSON.stringify({ ok: true, dataset }, null, 2)}\n`);
}

async function handleProxyRecord(flags) {
  const targetUrl = requiredFlag(flags, "target-url");
  const host = typeof flags.host === "string" && flags.host.trim()
    ? flags.host.trim()
    : "127.0.0.1";
  const port = readNumberFlag(flags.port, 5051, "port");
  const logFile = resolve(
    String(flags["log-file"] || join(ROOT_DIR, "artifacts", "proxy", "captured.jsonl")),
  );

  const server = createServer(async (request, response) => {
    try {
      await handleProxyRequest({ targetUrl, logFile, request, response });
    } catch (error) {
      response.statusCode = 500;
      response.setHeader("content-type", "application/json");
      response.end(
        JSON.stringify({
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  });

  await new Promise((resolveServer, rejectServer) => {
    server.once("error", rejectServer);
    server.listen(port, host, () => {
      server.off("error", rejectServer);
      resolveServer();
    });
  });

  process.stdout.write(
    `${JSON.stringify({
      ok: true,
      mode: "proxy-record",
      host,
      port,
      targetUrl,
      logFile,
    }, null, 2)}\n`,
  );
}

async function handleUploadProxyLog(flags) {
  const auth = await loadBackendAuthForDownload(flags);
  const endpointUrl = resolveConfig(
    flags["upload-url"],
    process.env.BURSTCHESTER_DEBUG_UPLOAD_URL,
    BURSTCHESTER_DEFAULTS.debugUploadUrl,
  );
  const filePath = requiredFlag(flags, "file");
  const content = await readFile(resolve(filePath), "utf8");
  const sourceModel = requiredFlag(flags, "source-model");
  const filename = typeof flags.filename === "string" && flags.filename.trim()
    ? flags.filename.trim()
    : basename(filePath);
  const metadata = buildProxyUploadMetadata({
    title:
      typeof flags.title === "string" && flags.title.trim()
        ? flags.title.trim()
        : `${sourceModel} proxy capture`,
    description: typeof flags.description === "string" ? flags.description : undefined,
    tags: typeof flags.tags === "string" ? flags.tags : undefined,
    taskType: typeof flags["task-type"] === "string" ? flags["task-type"] : undefined,
    sourceModel,
    baseModelHint: typeof flags["base-model-hint"] === "string" ? flags["base-model-hint"] : undefined,
  });

  const dataset = await uploadDebugDataset({
    endpointUrl,
    idToken: auth.bearerToken,
    filename,
    content,
    metadata: {
      language: typeof flags.language === "string" ? flags.language : undefined,
      license: typeof flags.license === "string" ? flags.license : undefined,
      outputModelId: typeof flags["output-model-id"] === "string" ? flags["output-model-id"] : undefined,
      pointCost: typeof flags["point-cost"] === "string" ? flags["point-cost"] : undefined,
      ...metadata,
    },
  });

  process.stdout.write(
    `${JSON.stringify({ ok: true, file: resolve(filePath), dataset }, null, 2)}\n`,
  );
}

async function handleUpdatePointCost(flags) {
  const session = await loadActiveSessionForBackendWrite(flags);
  const endpointUrl = resolveConfig(
    flags["update-url"],
    process.env.BURSTCHESTER_POINT_COST_UPDATE_URL,
    BURSTCHESTER_DEFAULTS.pointCostUpdateUrl,
  );
  const result = await updateAssetPointCost({
    endpointUrl,
    idToken: session.idToken,
    assetType: requiredFlag(flags, "asset-type"),
    assetId: requiredFlag(flags, "asset-id"),
    pointCost: requiredFlag(flags, "point-cost"),
  });

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

async function handleRegisterModel(flags) {
  const auth = await loadBackendAuthForDownload(flags);
  const endpointUrl = resolveConfig(
    flags["register-url"],
    process.env.BURSTCHESTER_MODEL_REGISTER_URL,
    BURSTCHESTER_DEFAULTS.modelRegisterUrl,
  );
  const datasetIds = hasDatasetSelectionFlags(flags)
    ? await resolveDatasetIdsInput({ flags, session: auth.session })
    : [];
  const model = await registerModel({
    endpointUrl,
    idToken: auth.bearerToken,
    huggingFaceUrl: requiredFlag(flags, "huggingface-url"),
    baseModel: typeof flags["base-model"] === "string" ? flags["base-model"] : "unknown",
    trainingDatasets: datasetIds,
    trainingMethod: typeof flags["training-method"] === "string" ? flags["training-method"] : "qlora",
    pointCost: typeof flags["point-cost"] === "string" ? flags["point-cost"] : undefined,
    ollamaPullUrl: typeof flags["ollama-pull-url"] === "string" ? flags["ollama-pull-url"] : "",
  });

  process.stdout.write(`${JSON.stringify({ ok: true, model }, null, 2)}\n`);
}

function hasDatasetSelectionFlags(flags) {
  return Boolean(
    typeof flags["dataset-id"] === "string"
      || typeof flags["dataset-file"] === "string"
      || flags["paste-dataset-list"] === true,
  );
}

async function handleTrain(flags) {
  const endpointUrl = resolveConfig(
    flags["backend-url"],
    process.env.BURSTCHESTER_BACKEND_URL,
    BURSTCHESTER_DEFAULTS.datasetDownloadUrl,
  );
  const session = await loadSession();
  const datasetIds = await resolveDatasetIdsInput({ flags, session });
  const datasetId = datasetIds[0];
  const modelRepo = requiredFlag(flags, "model-repo");
  const workspace = resolve(String(flags.workspace || join(ROOT_DIR, "artifacts", "training", datasetId)));
  const pythonBin = resolvePythonBin(flags);
  const trainingMethod = typeof flags["training-method"] === "string" ? flags["training-method"] : "qlora";

  const prepared = await prepareMergedDatasetForTraining({
    datasetIds,
    endpointUrl,
    authFlags: flags,
    preflightOnly: flags["preflight-only"] === true,
    workspace,
  });

  if (flags["preflight-only"] === true) {
    process.stdout.write(`${JSON.stringify({ ok: true, preflight: prepared.preflight }, null, 2)}\n`);
    return;
  }

  const manifest = buildTrainingManifest({
    datasetId,
    datasetIds,
    datasetPath: prepared.mergedDatasetPath,
    modelRepo,
    outputDir: join(workspace, "output"),
    trainingMethod,
    numTrainEpochs: flags.epochs,
    perDeviceTrainBatchSize: flags["batch-size"],
    gradientAccumulationSteps: flags["grad-accum"],
    learningRate: flags["learning-rate"],
    maxSeqLength: flags["max-seq-length"],
    loraRank: flags["lora-rank"],
    loraAlpha: flags["lora-alpha"],
    loraDropout: flags["lora-dropout"],
    loggingSteps: flags["logging-steps"],
    saveSteps: flags["save-steps"],
  });

  const result = await runTraining({
    manifest,
    workDir: workspace,
    pythonBin,
  });

  process.stdout.write(
    `${JSON.stringify({ datasetId, modelRepo, workspace, outputDir: manifest.outputDir, ...result }, null, 2)}\n`,
  );
}

async function handleTrainGemma4E2BFull(flags) {
  const endpointUrl = resolveConfig(
    flags["backend-url"],
    process.env.BURSTCHESTER_BACKEND_URL,
    BURSTCHESTER_DEFAULTS.datasetDownloadUrl,
  );
  const session = await loadSession();
  const datasetIds = await resolveDatasetIdsInput({ flags, session });
  const datasetId = datasetIds[0];
  const workspace = resolve(String(flags.workspace || join(ROOT_DIR, "artifacts", "training", `gemma4-e2b-full-${datasetId}`)));
  const pythonBin = resolvePythonBin(flags);
  const modelRepo = typeof flags["model-repo"] === "string" && flags["model-repo"].trim()
    ? flags["model-repo"].trim()
    : "google/gemma-4-E2B";

  const prepared = await prepareMergedDatasetForTraining({
    datasetIds,
    endpointUrl,
    authFlags: flags,
    preflightOnly: flags["preflight-only"] === true,
    workspace,
  });

  if (flags["preflight-only"] === true) {
    process.stdout.write(`${JSON.stringify({ ok: true, preflight: prepared.preflight }, null, 2)}\n`);
    return;
  }

  const manifest = buildGemma4E2BFullManifest({
    datasetId,
    datasetIds,
    datasetPath: prepared.mergedDatasetPath,
    modelRepo,
    outputDir: join(workspace, "output"),
    numTrainEpochs: flags.epochs,
    perDeviceTrainBatchSize: flags["batch-size"],
    gradientAccumulationSteps: flags["grad-accum"],
    learningRate: flags["learning-rate"],
    maxSeqLength: flags["max-seq-length"],
    loggingSteps: flags["logging-steps"],
    saveSteps: flags["save-steps"],
  });

  const result = await runTraining({
    manifest,
    workDir: workspace,
    pythonBin,
    scriptPath: defaultGemma4FullTrainerScriptPath(),
  });

  process.stdout.write(
    `${JSON.stringify({ datasetId, datasetIds, modelRepo: manifest.modelRepo, workspace, outputDir: manifest.outputDir, ...result }, null, 2)}\n`,
  );
}

async function handleTrainGemma2BItLora(flags) {
  const endpointUrl = resolveConfig(
    flags["backend-url"],
    process.env.BURSTCHESTER_BACKEND_URL,
    BURSTCHESTER_DEFAULTS.datasetDownloadUrl,
  );
  const session = await loadSession();
  const datasetIds = await resolveDatasetIdsInput({ flags, session });
  const datasetId = datasetIds[0];
  const workspace = resolve(String(flags.workspace || join(ROOT_DIR, "artifacts", "training", `gemma-2b-it-lora-${datasetId}`)));
  const pythonBin = resolvePythonBin(flags);
  const modelRepo = typeof flags["model-repo"] === "string" && flags["model-repo"].trim()
    ? flags["model-repo"].trim()
    : "google/gemma-2b-it";

  const prepared = await prepareMergedDatasetForTraining({
    datasetIds,
    endpointUrl,
    authFlags: flags,
    preflightOnly: flags["preflight-only"] === true,
    workspace,
  });

  if (flags["preflight-only"] === true) {
    process.stdout.write(`${JSON.stringify({ ok: true, preflight: prepared.preflight }, null, 2)}\n`);
    return;
  }

  const manifest = buildGemma2BItLoraManifest({
    datasetId,
    datasetIds,
    datasetPath: prepared.mergedDatasetPath,
    modelRepo,
    outputDir: join(workspace, "output"),
    numTrainEpochs: flags.epochs,
    perDeviceTrainBatchSize: flags["batch-size"],
    gradientAccumulationSteps: flags["grad-accum"],
    learningRate: flags["learning-rate"],
    maxSeqLength: flags["max-seq-length"],
    loraRank: flags["lora-rank"],
    loraAlpha: flags["lora-alpha"],
    loraDropout: flags["lora-dropout"],
    loggingSteps: flags["logging-steps"],
    saveSteps: flags["save-steps"],
  });

  const result = await runTraining({
    manifest,
    workDir: workspace,
    pythonBin,
    scriptPath: defaultGemma2BItLoraTrainerScriptPath(),
  });

  process.stdout.write(
    `${JSON.stringify({ datasetId, datasetIds, modelRepo: manifest.modelRepo, workspace, outputDir: manifest.outputDir, ...result }, null, 2)}\n`,
  );
}

function printUsage() {
  process.stdout.write(
    [
      "Burstchester CLI",
      "",
      "Commands:",
      "  auth status",
      "  auth huggingface [--token <hf_token>] [--clear]",
      "  auth profile --display-name <name> [--api-key <firebase-key>] [--profile-url <url>]",
      "  auth logout",
      "  access-token issue [--label <label>] [--token-url <url>]",
      "  dataset-list add --dataset-id <id>",
      "  dataset-list remove --dataset-id <id>",
      "  dataset-list show",
      "  dataset-list clear",
      "  dataset-list import --file <path>",
      "  dataset-list export --file <path>",
      "  download-dataset [--backend-url <url>] --dataset-id <id> [--access-token <token>] [--out-dir <dir>] [--extract false]",
      "  download-model --url <hf-url> [--access-token <token>] [--model-name <name>] [--out-dir <dir>]",
      "  download-model --repo <org/model> --file <filename> [--access-token <token>] [--revision <rev>] [--out-dir <dir>]",
      "  proxy-record --target-url <url> [--host <host>] [--port <port>] [--log-file <path>]",
      "  upload-test-dataset --file <path> [--title <title>] [--point-cost <points>] [--access-token <token>] [--upload-url <url>]",
      "  upload-proxy-log --file <path> --source-model <model> [--title <title>] [--point-cost <points>] [--access-token <token>] [--upload-url <url>]",
      "  register-model --huggingface-url <hf-url> [--base-model <name>] [--dataset-id <id> | --dataset-file <path> | --paste-dataset-list] [--training-method <method>] [--point-cost <points>] [--ollama-pull-url <url>] [--access-token <token>]",
      "  update-point-cost --asset-type <dataset|model> --asset-id <id> --point-cost <points>",
      "  train [--backend-url <url>] [--dataset-id <id> | --dataset-file <path> | --paste-dataset-list] --model-repo <org/model> [--access-token <token>] [--workspace <dir>] [--preflight-only]",
      "  train-gemma4-e2b-full [--backend-url <url>] [--dataset-id <id> | --dataset-file <path> | --paste-dataset-list] [--model-repo <org/model>] [--access-token <token>] [--workspace <dir>] [--preflight-only]",
      "  train-gemma-2b-it-lora [--backend-url <url>] [--dataset-id <id> | --dataset-file <path> | --paste-dataset-list] [--model-repo <org/model>] [--access-token <token>] [--workspace <dir>] [--preflight-only]",
      "",
    ].join("\n"),
  );
}

main(process.argv.slice(2)).catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});

function requiredConfig(flagValue, envValue, name, defaultValue = "") {
  const value = resolveConfig(flagValue, envValue, defaultValue);
  if (!value) {
    throw new Error(`Missing required flag or env for ${name}`);
  }

  return value;
}

function optionalConfig(flagValue, envValue, defaultValue = "") {
  const value = resolveConfig(flagValue, envValue, defaultValue);
  return value || null;
}

function resolveConfig(flagValue, envValue, defaultValue = "") {
  if (typeof flagValue === "string" && flagValue.trim()) {
    return flagValue.trim();
  }

  if (typeof envValue === "string" && envValue.trim()) {
    return envValue.trim();
  }

  if (typeof defaultValue === "string" && defaultValue.trim()) {
    return defaultValue.trim();
  }

  return "";
}

function resolveDownloadToken(flags, session) {
  return (
    normalizeLocalToken(typeof flags.token === "string" ? flags.token : "")
    || normalizeLocalToken(session?.huggingFaceToken)
    || normalizeLocalToken(process.env.HF_TOKEN)
    || normalizeLocalToken(process.env.HUGGING_FACE_HUB_TOKEN)
    || ""
  );
}

function normalizeLocalToken(token) {
  return typeof token === "string" && token.trim() ? token.trim() : null;
}

async function promptForToken(prompt) {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  try {
    return await rl.question(prompt);
  } finally {
    rl.close();
  }
}

async function prepareMergedDatasetForTraining({ datasetIds, endpointUrl, authFlags = {}, preflightOnly, workspace }) {
  const auth = await loadBackendAuthForDownload(authFlags);
  await ensureDir(workspace);
  const preflight = await preflightDatasetDownloads({
    endpointUrl,
    datasetIds,
    idToken: auth.bearerToken,
  });

  if (preflight.summary.failedCount > 0) {
    throw new Error(`Dataset preflight failed for: ${preflight.summary.failedDatasetIds.join(", ")}`);
  }

  if (preflightOnly) {
    return {
      preflight,
      mergedDatasetPath: null,
    };
  }

  const mergedParts = [];

  for (const currentDatasetId of datasetIds) {
    const metadata = await fetchDatasetPackageMetadata({
      endpointUrl,
      datasetId: currentDatasetId,
      idToken: auth.bearerToken,
    });

    const zipPath = join(workspace, `${currentDatasetId}.zip`);
    await downloadToFile({
      url: metadata.url,
      destination: zipPath,
    });

    const datasetDir = join(workspace, "datasets", currentDatasetId);
    const archive = await readFile(zipPath);
    await extractStoredZip(archive, datasetDir);
    mergedParts.push(join(datasetDir, "dataset.jsonl"));
  }

  return {
    preflight,
    mergedDatasetPath: await mergeTextFiles(
      mergedParts,
      join(workspace, "merged-dataset.jsonl"),
    ),
  };
}

async function loadActiveSessionForBackendWrite(flags) {
  let session = await loadSession();
  if (!session) {
    throw new Error("No CLI session found. Run `auth profile --display-name ...` first.");
  }

  const apiKey = resolveConfig(
    flags["api-key"],
    process.env.BURSTCHESTER_FIREBASE_API_KEY,
    BURSTCHESTER_DEFAULTS.firebaseConfig.apiKey,
  );
  if (isSessionExpired(session)) {
    session = {
      ...session,
      ...await refreshFirebaseSession({
        apiKey,
        refreshToken: session.refreshToken,
      }),
    };
    await saveSession(session);
  }

  return session;
}

async function handleProxyRequest({ targetUrl, logFile, request, response }) {
  const requestBuffer = await readIncomingRequestBody(request);
  const pathname = request.url ? new URL(request.url, "http://localhost").pathname : "/";
  const search = request.url ? new URL(request.url, "http://localhost").search : "";
  const contentType = String(request.headers["content-type"] || "");
  const parsedRequestBody = tryParseJsonBody(requestBuffer, contentType);

  if (isStreamingProxyRequest(parsedRequestBody)) {
    response.statusCode = 400;
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({
      ok: false,
      error: "Streaming requests are not supported by proxy-record. Send stream=false.",
    }));
    return;
  }

  const proxiedUrl = new URL(`${pathname}${search}`, ensureTrailingSlash(targetUrl));
  const normalizedRequestBody = normalizeProxyRequestBody(pathname, parsedRequestBody);
  const forwardHeaders = filterForwardHeaders(request.headers);
  let forwardBody = requestBuffer;

  if (normalizedRequestBody !== parsedRequestBody) {
    forwardBody = Buffer.from(JSON.stringify(normalizedRequestBody));
    forwardHeaders.set("content-type", "application/json");
    forwardHeaders.set("content-length", String(forwardBody.byteLength));
  }

  const upstream = await fetch(proxiedUrl, {
    method: request.method || "GET",
    headers: forwardHeaders,
    body: shouldSendBody(request.method) ? forwardBody : undefined,
  });
  const responseBuffer = Buffer.from(await upstream.arrayBuffer());
  const responseContentType = upstream.headers.get("content-type") || "";

  response.statusCode = upstream.status;
  upstream.headers.forEach((value, key) => {
    response.setHeader(key, value);
  });
  response.end(responseBuffer);

  if (!upstream.ok) {
    return;
  }

  const parsedResponseBody = tryParseJsonBody(responseBuffer, responseContentType);
  const sample = buildProxyLogSample({
    pathname,
    requestBody: normalizedRequestBody,
    responseBody: parsedResponseBody,
  });
  if (!sample) {
    return;
  }

  await appendProxyJsonlSample(logFile, sample);
}

async function readIncomingRequestBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks);
}

function tryParseJsonBody(buffer, contentType) {
  if (!String(contentType || "").toLowerCase().includes("application/json")) {
    return null;
  }

  const text = buffer.toString("utf8").trim();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function filterForwardHeaders(headers) {
  const nextHeaders = new Headers();
  for (const [key, rawValue] of Object.entries(headers)) {
    if (rawValue === undefined) {
      continue;
    }
    if (key.toLowerCase() === "host") {
      continue;
    }

    if (Array.isArray(rawValue)) {
      nextHeaders.set(key, rawValue.join(", "));
      continue;
    }

    nextHeaders.set(key, String(rawValue));
  }
  return nextHeaders;
}

function shouldSendBody(method) {
  return !["GET", "HEAD"].includes(String(method || "GET").toUpperCase());
}

function ensureTrailingSlash(url) {
  return url.endsWith("/") ? url : `${url}/`;
}

function readNumberFlag(rawValue, defaultValue, flagName) {
  if (rawValue === undefined || rawValue === null || rawValue === "") {
    return defaultValue;
  }

  const parsed = Number(rawValue);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid --${flagName}: ${rawValue}`);
  }

  return parsed;
}

function resolvePythonBin(flags) {
  if (typeof flags.python === "string" && flags.python.trim()) {
    return flags.python.trim();
  }

  const venvPython = join(ROOT_DIR, ".venv", "bin", "python");
  return existsSync(venvPython) ? venvPython : "python3";
}
