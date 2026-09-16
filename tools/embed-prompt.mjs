// Precompute the CLIP text embedding for the fixed cat prompt so the browser
// never has to download the 680 MB text encoder or run a tokenizer.
// Output: ../public/prompt_embeds.bin  (float32, shape [1, 77, 1024])
import { AutoTokenizer } from "@huggingface/transformers";
import ort from "./node_modules/@huggingface/transformers/node_modules/onnxruntime-node/dist/index.js";
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const BASE = "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main";
const PROMPT =
  process.argv[2] ??
  "a photograph of a cat, photorealistic, sharp focus, natural light, dslr, 50mm lens";

const here = path.dirname(fileURLToPath(import.meta.url));
const cacheDir = path.join(here, ".cache");
const outDir = path.join(here, "..", "public");
await mkdir(cacheDir, { recursive: true });
await mkdir(outDir, { recursive: true });

const encPath = path.join(cacheDir, "text_encoder.onnx");
if (!existsSync(encPath)) {
  console.log("downloading text encoder (~680 MB)...");
  const res = await fetch(`${BASE}/text_encoder/model.onnx`);
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  await writeFile(encPath, Buffer.from(await res.arrayBuffer()));
}

// SD 2.x / sd-turbo uses the OpenCLIP tokenizer: same BPE vocab as CLIP, pad id 0 ("!").
const tokenizer = await AutoTokenizer.from_pretrained("Xenova/clip-vit-base-patch16");
tokenizer.pad_token_id = 0;
const { input_ids } = await tokenizer(PROMPT, {
  padding: "max_length",
  max_length: 77,
  truncation: true,
  return_tensor: false,
});
console.log("tokens:", input_ids.length, input_ids.slice(0, 16).join(" "), "...");

const sess = await ort.InferenceSession.create(encPath, {
  executionProviders: ["cpu"],
});
const { last_hidden_state } = await sess.run({
  input_ids: new ort.Tensor("int32", Int32Array.from(input_ids), [1, 77]),
});
console.log("embedding dims:", last_hidden_state.dims, last_hidden_state.type);

let data = last_hidden_state.data;
if (last_hidden_state.type === "float16") {
  // convert fp16 -> fp32
  const u16 = data;
  const f32 = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) f32[i] = fp16ToF32(u16[i]);
  data = f32;
}
const out = path.join(outDir, "prompt_embeds.bin");
await writeFile(out, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
await writeFile(
  path.join(outDir, "prompt_embeds.json"),
  JSON.stringify({ prompt: PROMPT, dims: last_hidden_state.dims, dtype: "float32" }, null, 2)
);
console.log("wrote", out, data.byteLength, "bytes");

function fp16ToF32(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
