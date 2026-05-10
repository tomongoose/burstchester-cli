import test from "node:test";
import assert from "node:assert/strict";

import { summarizeDatasetPreflight } from "../src/lib/backend.mjs";

test("summarizeDatasetPreflight groups successes and failures", () => {
  const summary = summarizeDatasetPreflight([
    { datasetId: "legal-ko", ok: true, zipPath: "downloads/legal-ko/legal-ko.zip" },
    { datasetId: "finance-ko", ok: false, error: "not found" },
  ]);

  assert.deepEqual(summary.okDatasetIds, ["legal-ko"]);
  assert.deepEqual(summary.failedDatasetIds, ["finance-ko"]);
  assert.equal(summary.okCount, 1);
  assert.equal(summary.failedCount, 1);
});
