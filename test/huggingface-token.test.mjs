import test from "node:test";
import assert from "node:assert/strict";

import {
  normalizeHuggingFaceToken,
  resolveHuggingFaceToken,
} from "../src/lib/huggingface.mjs";

test("resolveHuggingFaceToken prefers explicit token over stored and env tokens", () => {
  const token = resolveHuggingFaceToken({
    explicitToken: "hf_explicit",
    storedToken: "hf_stored",
    envToken: "hf_env",
  });

  assert.equal(token, "hf_explicit");
});

test("resolveHuggingFaceToken falls back to stored token when explicit token is absent", () => {
  const token = resolveHuggingFaceToken({
    explicitToken: "",
    storedToken: "hf_stored",
    envToken: "hf_env",
  });

  assert.equal(token, "hf_stored");
});

test("normalizeHuggingFaceToken trims valid token and rejects blanks", () => {
  assert.equal(normalizeHuggingFaceToken("  hf_abc123  "), "hf_abc123");
  assert.equal(normalizeHuggingFaceToken("   "), null);
});
