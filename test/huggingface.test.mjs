import test from "node:test";
import assert from "node:assert/strict";

import { buildHuggingFaceFileUrl } from "../src/lib/huggingface.mjs";

test("buildHuggingFaceFileUrl builds resolve/main download URL", () => {
  const url = buildHuggingFaceFileUrl("burstchester/legal-ko-qlora", "adapter_model.safetensors");

  assert.equal(
    url,
    "https://huggingface.co/burstchester/legal-ko-qlora/resolve/main/adapter_model.safetensors",
  );
});
