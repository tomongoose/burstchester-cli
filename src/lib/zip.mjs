import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

export function readStoredZipEntries(buffer) {
  const entries = {};
  let offset = 0;

  while (offset + 30 <= buffer.length) {
    const signature = buffer.readUInt32LE(offset);
    if (signature !== 0x04034b50) {
      break;
    }

    const compressionMethod = buffer.readUInt16LE(offset + 8);
    if (compressionMethod !== 0) {
      throw new Error(`Unsupported ZIP compression method: ${compressionMethod}`);
    }

    const compressedSize = buffer.readUInt32LE(offset + 18);
    const uncompressedSize = buffer.readUInt32LE(offset + 22);
    const fileNameLength = buffer.readUInt16LE(offset + 26);
    const extraLength = buffer.readUInt16LE(offset + 28);

    const nameStart = offset + 30;
    const nameEnd = nameStart + fileNameLength;
    const dataStart = nameEnd + extraLength;
    const dataEnd = dataStart + compressedSize;
    const name = buffer.subarray(nameStart, nameEnd).toString("utf8");
    const data = buffer.subarray(dataStart, dataEnd);

    if (data.length !== uncompressedSize) {
      throw new Error(`Corrupted ZIP entry: ${name}`);
    }

    entries[name] = data;
    offset = dataEnd;
  }

  return entries;
}

export async function extractStoredZip(buffer, outDir) {
  const entries = readStoredZipEntries(buffer);

  for (const [name, data] of Object.entries(entries)) {
    const destination = join(outDir, name);
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, data);
  }

  return entries;
}
