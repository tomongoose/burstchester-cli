import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const DEFAULT_SESSION_PATH = join(homedir(), ".burstchester", "session.json");
const DEFAULT_ACCESS_TOKEN_PATH = join(homedir(), ".burstchester", "access-token");

export function getSessionPath(customPath) {
  return customPath || DEFAULT_SESSION_PATH;
}

export function getAccessTokenPath(customPath) {
  return customPath || DEFAULT_ACCESS_TOKEN_PATH;
}

export async function loadSession(customPath) {
  try {
    const raw = await readFile(getSessionPath(customPath), "utf8");
    return JSON.parse(raw);
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function loadAccessToken(customPath) {
  try {
    const raw = await readFile(getAccessTokenPath(customPath), "utf8");
    const token = raw.trim();
    return token || null;
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function saveSession(session, customPath) {
  const sessionPath = getSessionPath(customPath);
  await mkdir(dirname(sessionPath), { recursive: true });
  await writeFile(sessionPath, `${JSON.stringify(session, null, 2)}\n`, "utf8");
  return sessionPath;
}

export async function clearSession(customPath) {
  await rm(getSessionPath(customPath), { force: true });
}

export function normalizeDatasetId(datasetId) {
  if (typeof datasetId !== "string") {
    return null;
  }

  const trimmed = datasetId.trim();
  return trimmed ? trimmed : null;
}

export function addDatasetId(datasetIds = [], datasetId) {
  const normalized = normalizeDatasetId(datasetId);
  if (!normalized) {
    return [...datasetIds];
  }

  return datasetIds.includes(normalized) ? [...datasetIds] : [...datasetIds, normalized];
}

export function removeDatasetId(datasetIds = [], datasetId) {
  const normalized = normalizeDatasetId(datasetId);
  if (!normalized) {
    return [...datasetIds];
  }

  return datasetIds.filter((value) => value !== normalized);
}

export function clearDatasetIds(_datasetIds = []) {
  return [];
}
