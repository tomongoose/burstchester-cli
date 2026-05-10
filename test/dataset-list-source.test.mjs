import test from "node:test";
import assert from "node:assert/strict";

import { resolveDatasetIdsInput } from "../src/lib/dataset-list-source.mjs";

test("resolveDatasetIdsInput reads newline-delimited dataset ids from stdin", async () => {
  const ids = await resolveDatasetIdsInput({
    flags: {
      "paste-dataset-list": true,
    },
    session: {},
    readStdinImpl: async () => "dataset-1\ndataset-2\n\ndataset-1\n",
  });

  assert.deepEqual(ids, ["dataset-1", "dataset-2"]);
});

test("resolveDatasetIdsInput prefers an explicit dataset id over pasted input", async () => {
  const ids = await resolveDatasetIdsInput({
    flags: {
      "dataset-id": "dataset-explicit",
      "paste-dataset-list": true,
    },
    session: {
      datasetIds: ["dataset-stored"],
    },
    readStdinImpl: async () => "dataset-1\ndataset-2\n",
  });

  assert.deepEqual(ids, ["dataset-explicit"]);
});
