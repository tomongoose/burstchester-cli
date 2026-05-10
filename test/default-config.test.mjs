import test from "node:test";
import assert from "node:assert/strict";

import { BURSTCHESTER_DEFAULTS } from "../src/lib/default-config.mjs";

test("BURSTCHESTER_DEFAULTS embeds the shared Firebase project config", () => {
  assert.equal(
    BURSTCHESTER_DEFAULTS.firebaseConfig.apiKey,
    "AIzaSyBT48mVt9IDw6Ctf_VjNl0JNc4S1SrVZfs",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.healthCheckUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/healthCheck",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.profileUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/upsertCliProfile",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.datasetDownloadUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/prepareDatasetDownload",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.modelDownloadUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/recordModelDownload",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.modelRegisterUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/registerModelHttp",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.pointCostUpdateUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/updateAssetPointCost",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.accessTokenUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/issueAccessToken",
  );
  assert.equal(
    BURSTCHESTER_DEFAULTS.debugUploadUrl,
    "https://us-central1-bustchester-e08c3.cloudfunctions.net/debugUploadDataset",
  );
});
