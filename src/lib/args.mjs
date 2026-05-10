export function parseArgs(argv) {
  const [command, ...rest] = argv;
  const flags = {};
  const positionals = [];

  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (!token.startsWith("--")) {
      positionals.push(token);
      continue;
    }

    const key = token.slice(2);
    const next = rest[index + 1];
    if (!next || next.startsWith("--")) {
      flags[key] = true;
      continue;
    }

    flags[key] = next;
    index += 1;
  }

  return {
    command,
    flags,
    positionals,
  };
}

export function requiredFlag(flags, key) {
  const value = flags[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Missing required flag --${key}`);
  }

  return value.trim();
}
