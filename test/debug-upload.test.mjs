import test from "node:test";
import assert from "node:assert/strict";

import { buildDebugUploadRequest } from "../src/lib/backend.mjs";

test("buildDebugUploadRequest includes bearer token and dataset payload", () => {
  const request = buildDebugUploadRequest({
    endpointUrl: "https://functions.example/debugUploadDataset",
    idToken: "firebase-id-token",
    filename: "legal-ko.jsonl",
    content: '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n',
    metadata: {
      title: "Legal Debug Dataset",
      sourceModel: "human",
    },
  });

  assert.equal(request.url, "https://functions.example/debugUploadDataset");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers.authorization, "Bearer firebase-id-token");
  assert.deepEqual(JSON.parse(request.options.body), {
    filename: "legal-ko.jsonl",
    content: '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n',
    title: "Legal Debug Dataset",
    sourceModel: "human",
  });
});
