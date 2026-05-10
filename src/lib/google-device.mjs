export function buildCliGoogleAuthRequest({
  authUrl,
  firebaseIdToken,
  action,
  deviceCode,
}) {
  return {
    url: authUrl,
    options: {
      method: "POST",
      headers: {
        authorization: `Bearer ${firebaseIdToken}`,
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({
        action,
        ...(deviceCode ? { deviceCode } : {}),
      }),
    },
  };
}

export async function startGoogleDeviceFlow({
  authUrl,
  firebaseIdToken,
  fetchImpl = fetch,
}) {
  const request = buildCliGoogleAuthRequest({
    authUrl,
    firebaseIdToken,
    action: "start",
  });
  const response = await fetchImpl(request.url, request.options);
  const payload = await response.json();
  if (!response.ok || payload?.ok !== true) {
    throw new Error(payload?.error || "Failed to start Google device flow.");
  }

  return payload;
}

export async function pollGoogleDeviceFlow({
  authUrl,
  firebaseIdToken,
  deviceCode,
  interval,
  fetchImpl = fetch,
  sleepImpl = sleep,
}) {
  let pollIntervalMs = Number(interval || 5) * 1000;

  for (;;) {
    await sleepImpl(pollIntervalMs);

    const request = buildCliGoogleAuthRequest({
      authUrl,
      firebaseIdToken,
      action: "poll",
      deviceCode,
    });
    const response = await fetchImpl(request.url, request.options);
    const payload = await response.json();
    if (!response.ok || payload?.ok !== true) {
      throw new Error(payload?.error || "Google device flow failed.");
    }

    if (payload.status === "approved" && payload.idToken) {
      return {
        id_token: payload.idToken,
      };
    }

    if (payload.status === "pending") {
      pollIntervalMs = Number(payload.interval || 5) * 1000;
      continue;
    }

    throw new Error("Unexpected Google device flow response.");
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
