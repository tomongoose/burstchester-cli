export function buildDatasetDownloadUrl(endpointUrl, datasetId) {
  const url = new URL(endpointUrl);
  url.searchParams.set("datasetId", datasetId);
  return url.toString();
}

export function summarizeDatasetPreflight(results) {
  const okDatasetIds = results.filter((result) => result.ok).map((result) => result.datasetId);
  const failedDatasetIds = results.filter((result) => !result.ok).map((result) => result.datasetId);

  return {
    okDatasetIds,
    failedDatasetIds,
    okCount: okDatasetIds.length,
    failedCount: failedDatasetIds.length,
  };
}

export function buildDebugUploadRequest({
  endpointUrl,
  idToken,
  filename,
  content,
  metadata = {},
}) {
  return {
    url: endpointUrl,
    options: {
      method: "POST",
      headers: {
        authorization: `Bearer ${idToken}`,
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({
        filename,
        content,
        ...metadata,
      }),
    },
  };
}

export function buildIssueAccessTokenRequest({
  endpointUrl,
  idToken,
  label,
}) {
  return {
    url: endpointUrl,
    options: {
      method: "POST",
      headers: {
        authorization: `Bearer ${idToken}`,
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({ label }),
    },
  };
}

export async function issueAccessToken({
  endpointUrl,
  idToken,
  label,
  fetchImpl = fetch,
}) {
  const request = buildIssueAccessTokenRequest({
    endpointUrl,
    idToken,
    label,
  });
  const response = await fetchImpl(request.url, request.options);
  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Access token issue failed with status ${response.status}.`);
  }
  return payload;
}

export async function fetchDatasetPackageMetadata({
  endpointUrl,
  datasetId,
  idToken = "",
  fetchImpl = fetch,
}) {
  const headers = {
    accept: "application/json",
  };
  if (idToken) {
    headers.authorization = `Bearer ${idToken}`;
  }

  const response = await fetchImpl(buildDatasetDownloadUrl(endpointUrl, datasetId), {
    method: "GET",
    headers,
  });

  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Dataset download request failed with status ${response.status}.`);
  }

  if (typeof payload.url !== "string" || !payload.url) {
    throw new Error("Dataset download response did not include a signed url.");
  }

  return payload;
}

export function buildRecordModelDownloadRequest({
  endpointUrl,
  idToken,
  modelName,
  sourceUrl,
}) {
  return {
    url: endpointUrl,
    options: {
      method: "POST",
      headers: {
        authorization: `Bearer ${idToken}`,
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({
        modelName,
        sourceUrl,
      }),
    },
  };
}

export async function recordModelDownload({
  endpointUrl,
  idToken,
  modelName,
  sourceUrl,
  fetchImpl = fetch,
}) {
  const request = buildRecordModelDownloadRequest({
    endpointUrl,
    idToken,
    modelName,
    sourceUrl,
  });
  const response = await fetchImpl(request.url, request.options);
  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Model download recording failed with status ${response.status}.`);
  }
  return payload;
}

export function buildRegisterModelRequest({
  endpointUrl,
  idToken,
  huggingFaceUrl,
  baseModel,
  trainingDatasets = [],
  trainingMethod,
  pointCost,
  ollamaPullUrl,
}) {
  return {
    url: endpointUrl,
    options: {
      method: "POST",
      headers: {
        authorization: `Bearer ${idToken}`,
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({
        huggingFaceUrl,
        baseModel,
        trainingDatasets,
        trainingMethod,
        pointCost,
        ollamaPullUrl,
      }),
    },
  };
}

export async function registerModel({
  endpointUrl,
  idToken,
  huggingFaceUrl,
  baseModel,
  trainingDatasets = [],
  trainingMethod,
  pointCost,
  ollamaPullUrl,
  fetchImpl = fetch,
}) {
  const request = buildRegisterModelRequest({
    endpointUrl,
    idToken,
    huggingFaceUrl,
    baseModel,
    trainingDatasets,
    trainingMethod,
    pointCost,
    ollamaPullUrl,
  });
  const response = await fetchImpl(request.url, request.options);
  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Model registration failed with status ${response.status}.`);
  }
  return payload.model;
}

export function buildUpdateAssetPointCostRequest({
  endpointUrl,
  idToken,
  assetType,
  assetId,
  pointCost,
}) {
  return {
    url: endpointUrl,
    options: {
      method: "POST",
      headers: {
        authorization: `Bearer ${idToken}`,
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({
        assetType,
        assetId,
        pointCost,
      }),
    },
  };
}

export async function updateAssetPointCost({
  endpointUrl,
  idToken,
  assetType,
  assetId,
  pointCost,
  fetchImpl = fetch,
}) {
  const request = buildUpdateAssetPointCostRequest({
    endpointUrl,
    idToken,
    assetType,
    assetId,
    pointCost,
  });
  const response = await fetchImpl(request.url, request.options);
  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Point cost update failed with status ${response.status}.`);
  }
  return payload;
}

export async function uploadDebugDataset({
  endpointUrl,
  idToken,
  filename,
  content,
  metadata = {},
  fetchImpl = fetch,
}) {
  const request = buildDebugUploadRequest({
    endpointUrl,
    idToken,
    filename,
    content,
    metadata,
  });

  const response = await fetchImpl(request.url, request.options);
  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Debug dataset upload failed with status ${response.status}.`);
  }

  return payload.dataset;
}

export async function preflightDatasetDownloads({
  endpointUrl,
  datasetIds,
  idToken = "",
  fetchImpl = fetch,
}) {
  const results = [];

  for (const datasetId of datasetIds) {
    try {
      const payload = await fetchDatasetPackageMetadata({
        endpointUrl,
        datasetId,
        idToken,
        fetchImpl,
      });
      results.push({
        datasetId,
        ok: true,
        zipPath: payload.zipPath,
        cached: payload.cached,
      });
    } catch (error) {
      results.push({
        datasetId,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return {
    results,
    summary: summarizeDatasetPreflight(results),
  };
}
