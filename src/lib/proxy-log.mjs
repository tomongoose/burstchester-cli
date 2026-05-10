import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

const DATASET_DESCRIPTION = "Recorded via Burstchester CLI proxy.";

export function buildProxyLogSample({
  pathname,
  requestBody,
  responseBody,
}) {
  const normalizedPath = String(pathname || "").trim();

  if (normalizedPath.endsWith("/v1/chat/completions")) {
    return buildChatSample(
      normalizeMessagesArray(requestBody?.messages),
      normalizeSingleMessage(responseBody?.choices?.[0]?.message, "assistant"),
    );
  }

  if (normalizedPath.endsWith("/api/chat")) {
    return buildChatSample(
      normalizeMessagesArray(requestBody?.messages),
      normalizeSingleMessage(responseBody?.message, "assistant"),
    );
  }

  if (normalizedPath.endsWith("/api/generate")) {
    return buildPromptSample(
      requestBody?.prompt,
      responseBody?.response,
    );
  }

  if (normalizedPath.endsWith("/v1/completions")) {
    return buildPromptSample(
      requestBody?.prompt,
      responseBody?.choices?.[0]?.text,
    );
  }

  if (normalizedPath.endsWith("/v1/responses")) {
    return buildPromptSample(
      extractResponsesInputText(requestBody?.input),
      extractResponsesOutputText(responseBody),
    );
  }

  return null;
}

export async function appendProxyJsonlSample(logFile, sample) {
  const line = `${JSON.stringify(sample)}\n`;
  await mkdir(dirname(logFile), { recursive: true });
  await appendFile(logFile, line, "utf8");
}

export function buildProxyUploadMetadata({
  title,
  description = DATASET_DESCRIPTION,
  tags = "proxy-log,llm-output",
  taskType = "chat",
  sourceModel,
  baseModelHint,
}) {
  if (typeof sourceModel !== "string" || !sourceModel.trim()) {
    throw new Error("sourceModel is required.");
  }

  const normalizedSourceModel = sourceModel.trim();
  return {
    title: String(title || "").trim() || "Proxy capture",
    description: String(description || "").trim() || DATASET_DESCRIPTION,
    tags: String(tags || "").trim() || "proxy-log,llm-output",
    taskType: String(taskType || "").trim() || "chat",
    sourceModel: normalizedSourceModel,
    baseModelHint:
      typeof baseModelHint === "string" && baseModelHint.trim()
        ? baseModelHint.trim()
        : normalizedSourceModel,
  };
}

export function isStreamingProxyRequest(body) {
  return Boolean(body && typeof body === "object" && body.stream === true);
}

export function normalizeProxyRequestBody(pathname, body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return body;
  }

  if (
    (String(pathname).endsWith("/api/chat") || String(pathname).endsWith("/api/generate"))
    && body.stream !== false
  ) {
    return {
      ...body,
      stream: false,
    };
  }

  return body;
}

function buildChatSample(messages, assistantMessage) {
  if (!Array.isArray(messages) || messages.length === 0 || !assistantMessage) {
    return null;
  }

  return {
    messages: [...messages, assistantMessage],
  };
}

function buildPromptSample(prompt, responseText) {
  const userContent = normalizeText(prompt);
  const assistantContent = normalizeText(responseText);
  if (!userContent || !assistantContent) {
    return null;
  }

  return {
    messages: [
      { role: "user", content: userContent },
      { role: "assistant", content: assistantContent },
    ],
  };
}

function normalizeMessagesArray(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages
    .map((message) => normalizeSingleMessage(message))
    .filter(Boolean);
}

function normalizeSingleMessage(message, fallbackRole = "user") {
  if (!message || typeof message !== "object") {
    return null;
  }

  const role = normalizeRole(message.role, fallbackRole);
  const content = extractMessageContent(message.content);
  if (!role || !content) {
    return null;
  }

  return { role, content };
}

function normalizeRole(role, fallbackRole) {
  const normalized = String(role || fallbackRole).trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  return normalized;
}

function extractMessageContent(content) {
  if (typeof content === "string") {
    return normalizeText(content);
  }

  if (!Array.isArray(content)) {
    return normalizeText(content);
  }

  const joined = content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }
      if (part && typeof part === "object" && typeof part.text === "string") {
        return part.text;
      }
      if (part && typeof part === "object" && typeof part.input_text === "string") {
        return part.input_text;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");

  return normalizeText(joined);
}

function extractResponsesInputText(input) {
  if (typeof input === "string") {
    return input;
  }

  if (Array.isArray(input)) {
    return input
      .map((entry) => {
        if (typeof entry === "string") {
          return entry;
        }
        if (entry && typeof entry === "object" && Array.isArray(entry.content)) {
          return entry.content
            .map((part) => {
              if (part && typeof part === "object" && typeof part.text === "string") {
                return part.text;
              }
              if (part && typeof part === "object" && typeof part.input_text === "string") {
                return part.input_text;
              }
              return "";
            })
            .filter(Boolean)
            .join("\n");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }

  return normalizeText(input);
}

function extractResponsesOutputText(responseBody) {
  if (typeof responseBody?.output_text === "string") {
    return responseBody.output_text;
  }

  if (!Array.isArray(responseBody?.output)) {
    return "";
  }

  return responseBody.output
    .map((entry) => {
      if (!entry || typeof entry !== "object" || !Array.isArray(entry.content)) {
        return "";
      }
      return entry.content
        .map((part) => {
          if (part && typeof part === "object" && typeof part.text === "string") {
            return part.text;
          }
          if (part && typeof part === "object" && typeof part.output_text === "string") {
            return part.output_text;
          }
          return "";
        })
        .filter(Boolean)
        .join("\n");
    })
    .filter(Boolean)
    .join("\n");
}

function normalizeText(value) {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || "";
  }

  if (value === null || value === undefined) {
    return "";
  }

  const trimmed = String(value).trim();
  return trimmed || "";
}
