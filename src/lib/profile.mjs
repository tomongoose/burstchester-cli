export async function upsertCliProfile({
  profileUrl,
  idToken,
  displayName,
  photoURL,
  fetchImpl = fetch,
}) {
  const response = await fetchImpl(profileUrl, {
    method: "POST",
    headers: {
      authorization: `Bearer ${idToken}`,
      "content-type": "application/json",
      accept: "application/json",
    },
    body: JSON.stringify({
      displayName,
      photoURL,
    }),
  });

  const payload = await response.json();
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Profile upsert failed with status ${response.status}.`);
  }

  return payload.profile;
}
