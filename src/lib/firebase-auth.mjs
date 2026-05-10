const FIREBASE_AUTH_BASE_URL = "https://identitytoolkit.googleapis.com/v1";
const FIREBASE_SECURE_TOKEN_URL = "https://securetoken.googleapis.com/v1/token";

export function buildAnonymousSignInRequest(apiKey) {
  return {
    url: `${FIREBASE_AUTH_BASE_URL}/accounts:signUp?key=${encodeURIComponent(apiKey)}`,
    method: "POST",
    body: {
      returnSecureToken: true,
    },
  };
}

export function buildGoogleLinkRequest({ apiKey, firebaseIdToken, googleIdToken }) {
  return {
    url: `${FIREBASE_AUTH_BASE_URL}/accounts:signInWithIdp?key=${encodeURIComponent(apiKey)}`,
    method: "POST",
    body: {
      postBody: `id_token=${encodeURIComponent(googleIdToken)}&providerId=google.com`,
      requestUri: "http://localhost",
      idToken: firebaseIdToken,
      returnIdpCredential: true,
      returnSecureToken: true,
    },
  };
}

export function decodeJwtPayload(jwt) {
  const [, payload] = String(jwt).split(".");
  if (!payload) {
    throw new Error("Invalid JWT payload.");
  }

  return JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
}

export async function signInAnonymously({ apiKey, fetchImpl = fetch }) {
  const request = buildAnonymousSignInRequest(apiKey);
  const payload = await postJson(fetchImpl, request);
  return normalizeFirebaseSession(payload, {
    isAnonymous: true,
    providerId: "anonymous",
  });
}

export async function refreshFirebaseSession({ apiKey, refreshToken, fetchImpl = fetch }) {
  const response = await fetchImpl(
    `${FIREBASE_SECURE_TOKEN_URL}?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: {
        "content-type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        grant_type: "refresh_token",
        refresh_token: refreshToken,
      }),
    },
  );

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || "Failed to refresh Firebase session.");
  }

  return {
    userId: payload.user_id,
    idToken: payload.id_token,
    refreshToken: payload.refresh_token,
    expiresAt: Date.now() + Number(payload.expires_in || 0) * 1000,
  };
}

export async function linkGoogleAccount({
  apiKey,
  firebaseIdToken,
  googleIdToken,
  fetchImpl = fetch,
}) {
  const request = buildGoogleLinkRequest({
    apiKey,
    firebaseIdToken,
    googleIdToken,
  });
  const payload = await postJson(fetchImpl, request);
  return normalizeFirebaseSession(payload, {
    isAnonymous: false,
    providerId: payload.providerId || "google.com",
  });
}

export function isSessionExpired(session, now = Date.now()) {
  return !session?.expiresAt || Number(session.expiresAt) - now <= 60_000;
}

function normalizeFirebaseSession(payload, defaults = {}) {
  return {
    userId: payload.localId || payload.user_id,
    idToken: payload.idToken || payload.id_token,
    refreshToken: payload.refreshToken || payload.refresh_token,
    expiresAt: Date.now() + Number(payload.expiresIn || payload.expires_in || 0) * 1000,
    isAnonymous: defaults.isAnonymous ?? !payload.email,
    providerId: defaults.providerId ?? payload.providerId ?? "unknown",
    email: payload.email || "",
  };
}

async function postJson(fetchImpl, request) {
  const response = await fetchImpl(request.url, {
    method: request.method,
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(request.body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || "Firebase auth request failed.");
  }

  return payload;
}
