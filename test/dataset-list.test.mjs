import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  addDatasetId,
  clearDatasetIds,
  loadAccessToken,
  normalizeDatasetId,
  removeDatasetId,
} from "../src/lib/session.mjs";

test("normalizeDatasetId trims values and rejects blanks", () => {
  assert.equal(normalizeDatasetId("  legal-ko  "), "legal-ko");
  assert.equal(normalizeDatasetId("   "), null);
});

test("addDatasetId appends unique dataset ids", () => {
  const ids = addDatasetId(["dataset-1"], " dataset-2 ");

  assert.deepEqual(ids, ["dataset-1", "dataset-2"]);
  assert.deepEqual(addDatasetId(ids, "dataset-2"), ["dataset-1", "dataset-2"]);
});

test("removeDatasetId and clearDatasetIds update the list predictably", () => {
  const ids = ["dataset-1", "dataset-2", "dataset-3"];

  assert.deepEqual(removeDatasetId(ids, "dataset-2"), ["dataset-1", "dataset-3"]);
  assert.deepEqual(clearDatasetIds(ids), []);
});

test("loadAccessToken reads a saved CLI access token", async () => {
  const dir = await mkdtemp(join(tmpdir(), "burstchester-token-"));
  const tokenPath = join(dir, "access-token");

  try {
    await writeFile(tokenPath, "  bst_uid_token_secret\n", "utf8");

    assert.equal(await loadAccessToken(tokenPath), "bst_uid_token_secret");
    assert.equal(await loadAccessToken(join(dir, "missing")), null);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
