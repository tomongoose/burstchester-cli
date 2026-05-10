import test from "node:test";
import assert from "node:assert/strict";

import { buildCliGoogleAuthRequest } from "../src/lib/google-device.mjs";

test("buildCliGoogleAuthRequest builds authenticated backend payload for start", () => {
  const request = buildCliGoogleAuthRequest({
    authUrl: "https://functions.example/cliGoogleAuth",
    firebaseIdToken: "firebase-id-token",
    action: "start",
  });

  assert.equal(request.url, "https://functions.example/cliGoogleAuth");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers.authorization, "Bearer firebase-id-token");
  assert.deepEqual(JSON.parse(request.options.body), {
    action: "start",
  });
});

test("buildCliGoogleAuthRequest includes deviceCode when polling", () => {
  const request = buildCliGoogleAuthRequest({
    authUrl: "https://functions.example/cliGoogleAuth",
    firebaseIdToken: "firebase-id-token",
    action: "poll",
    deviceCode: "device-code",
  });

  assert.deepEqual(JSON.parse(request.options.body), {
    action: "poll",
    deviceCode: "device-code",
  });
});
