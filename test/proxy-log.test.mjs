import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  appendProxyJsonlSample,
  buildProxyLogSample,
  buildProxyUploadMetadata,
} from "../src/lib/proxy-log.mjs";

test("buildProxyLogSample converts OpenAI chat completion exchange into dataset sample", () => {
  const sample = buildProxyLogSample({
    pathname: "/v1/chat/completions",
    requestBody: {
      model: "gemma3:12b",
      messages: [
        { role: "system", content: "You are precise." },
        { role: "user", content: "Summarize this clause." },
      ],
    },
    responseBody: {
      choices: [
        {
          message: {
            role: "assistant",
            content: "The clause limits liability to direct damages.",
          },
        },
      ],
    },
  });

  assert.deepEqual(sample, {
    messages: [
      { role: "system", content: "You are precise." },
      { role: "user", content: "Summarize this clause." },
      {
        role: "assistant",
        content: "The clause limits liability to direct damages.",
      },
    ],
  });
});

test("buildProxyLogSample converts Ollama generate exchange into dataset sample", () => {
  const sample = buildProxyLogSample({
    pathname: "/api/generate",
    requestBody: {
      model: "gemma3:12b",
      prompt: "Write a short answer about indemnity.",
    },
    responseBody: {
      response: "Indemnity shifts specified losses from one party to another.",
    },
  });

  assert.deepEqual(sample, {
    messages: [
      { role: "user", content: "Write a short answer about indemnity." },
      {
        role: "assistant",
        content: "Indemnity shifts specified losses from one party to another.",
      },
    ],
  });
});

test("appendProxyJsonlSample writes newline-delimited dataset samples", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "burstchester-proxy-log-"));
  const logFile = join(tempDir, "proxy.jsonl");

  await appendProxyJsonlSample(logFile, {
    messages: [
      { role: "user", content: "q1" },
      { role: "assistant", content: "a1" },
    ],
  });
  await appendProxyJsonlSample(logFile, {
    messages: [
      { role: "user", content: "q2" },
      { role: "assistant", content: "a2" },
    ],
  });

  const content = await readFile(logFile, "utf8");
  assert.equal(
    content,
    '{"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"}]}\n'
      + '{"messages":[{"role":"user","content":"q2"},{"role":"assistant","content":"a2"}]}\n',
  );
});

test("buildProxyUploadMetadata supplies proxy-friendly dataset defaults", () => {
  const metadata = buildProxyUploadMetadata({
    title: "Gemma proxy capture",
    sourceModel: "gemma3:12b",
  });

  assert.deepEqual(metadata, {
    title: "Gemma proxy capture",
    description: "Recorded via Burstchester CLI proxy.",
    tags: "proxy-log,llm-output",
    taskType: "chat",
    sourceModel: "gemma3:12b",
    baseModelHint: "gemma3:12b",
  });
});
