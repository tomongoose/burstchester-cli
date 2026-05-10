import { basename, join } from "node:path";

import { downloadToFile, ensureDir } from "./download.mjs";

export function buildHuggingFaceFileUrl(repo, file, revision = "main") {
  return `https://huggingface.co/${repo}/resolve/${revision}/${file}`;
}

export function normalizeHuggingFaceToken(token) {
  if (typeof token !== "string") {
    return null;
  }

  const trimmed = token.trim();
  return trimmed ? trimmed : null;
}

export function resolveHuggingFaceToken({
  explicitToken,
  storedToken,
  envToken,
}) {
  return (
    normalizeHuggingFaceToken(explicitToken)
    || normalizeHuggingFaceToken(storedToken)
    || normalizeHuggingFaceToken(envToken)
    || ""
  );
}

export async function downloadHuggingFaceFile({
  url,
  repo,
  file,
  revision = "main",
  outDir,
  token = "",
  fetchImpl = fetch,
}) {
  const resolvedUrl = url || buildHuggingFaceFileUrl(repo, file, revision);
  const filename = basename(new URL(resolvedUrl).pathname) || file || "model.bin";
  await ensureDir(outDir);

  const headers = token ? { authorization: `Bearer ${token}` } : {};
  const destination = join(outDir, filename);
  await downloadToFile({
    url: resolvedUrl,
    destination,
    headers,
    fetchImpl,
  });

  return {
    url: resolvedUrl,
    path: destination,
  };
}
