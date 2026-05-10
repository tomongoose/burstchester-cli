import test from "node:test";
import assert from "node:assert/strict";

import {
  buildAnonymousSignInRequest,
  buildGoogleLinkRequest,
  decodeJwtPayload,
} from "../src/lib/firebase-auth.mjs";

test("buildAnonymousSignInRequest points at Firebase anonymous signup endpoint", () => {
  const request = buildAnonymousSignInRequest("firebase-api-key");

  assert.equal(
    request.url,
    "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=firebase-api-key",
  );
  assert.equal(request.method, "POST");
  assert.deepEqual(request.body, {
    returnSecureToken: true,
  });
});

test("buildGoogleLinkRequest includes current firebase token and google id token", () => {
  const request = buildGoogleLinkRequest({
    apiKey: "firebase-api-key",
    firebaseIdToken: "firebase-id-token",
    googleIdToken: "google-id-token",
  });

  assert.equal(
    request.url,
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key=firebase-api-key",
  );
  assert.equal(request.body.idToken, "firebase-id-token");
  assert.match(request.body.postBody, /id_token=google-id-token/);
  assert.match(request.body.postBody, /providerId=google\.com/);
});

test("decodeJwtPayload returns JSON payload from JWT body", () => {
  const payload = {
    email: "alice@example.com",
    name: "Alice",
    picture: "https://example.com/avatar.png",
  };
  const encoded = Buffer.from(JSON.stringify(payload))
    .toString("base64url");
  const jwt = `header.${encoded}.signature`;

  assert.deepEqual(decodeJwtPayload(jwt), payload);
});
