import { normalizeDatasetId } from "./session.mjs";

export function parseDatasetIdFile(text) {
  const ids = [];

  for (const line of String(text).split(/\r?\n/)) {
    const normalized = normalizeDatasetId(line);
    if (normalized && !ids.includes(normalized)) {
      ids.push(normalized);
    }
  }

  return ids;
}

export function serializeDatasetIds(datasetIds) {
  const normalized = [];

  for (const value of datasetIds) {
    const datasetId = normalizeDatasetId(value);
    if (datasetId && !normalized.includes(datasetId)) {
      normalized.push(datasetId);
    }
  }

  return normalized.length > 0 ? `${normalized.join("\n")}\n` : "";
}
