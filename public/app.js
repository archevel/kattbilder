// Kattbilder: cat images generated in the browser with onnxruntime-web on WebGPU.
//
// Two pipelines:
//   sdturbo  SD-Turbo fp16 (1.75 GB), one denoising step, 512x512, fixed cat prompt.
//            Needs WebGPU "shader-f16". Best quality.
//   ddpm     google/ddpm-ema-cat-256 (455 MB fp32), unconditional, cats only by construction,
//            DDIM sampling with N steps, 256x256. Works without shader-f16 (e.g. Chrome/Linux/NVIDIA).
// Selection: ?model=sdturbo|ddpm, default sdturbo if shader-f16 is available, otherwise ddpm.
// An fp32 SD-Turbo variant exists (?model=sdturbo&precision=fp32) but needs >8 GB of GPU memory.

const params = new URLSearchParams(location.search);
const BATCH = Number(params.get("batch")) || 10;      // finished cats kept queued ahead of the viewer
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const UNET_OPT = { freeDimensionOverrides: { batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77 } };
const VAE_OPT = { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } };
const MODELS = {
  sdturbo_fp16: {
    base: "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main",
    unet: { url: "unet/model.onnx", sizeMB: 1653, opt: UNET_OPT },
    vae_decoder: { url: "vae_decoder/model.onnx", sizeMB: 95, opt: VAE_OPT },
  },
  sdturbo_fp32: {
    base: "models/sdturbo-fp32",   // produced by tools/fp16_to_fp32.py + tools/split_external_data.py
    unet: {
      url: "unet.onnx", sizeMB: 2, opt: UNET_OPT,
      external: [
        { url: "unet_0.bin", sizeMB: 793 }, { url: "unet_1.bin", sizeMB: 897 },
        { url: "unet_2.bin", sizeMB: 852 }, { url: "unet_3.bin", sizeMB: 761 },
      ],
    },
    vae_decoder: { url: "vae_decoder.onnx", sizeMB: 189, opt: VAE_OPT },
  },
  ddpm: {
    base: "https://huggingface.co/archevel/kattomat/resolve/main/ddpm-cat",   // exported by tools/export_ddpm_cat.py
    unet: { url: "unet.onnx", sizeMB: 434, opt: {} },
  },
};

const $ = (id) => document.getElementById(id);
const els = {
  canvas: $("cat"), overlay: $("overlay"), status: $("status"), progress: $("progress"),
  bar: $("bar"), next: $("next"), queue: $("queue"), log: $("log"), sub: $("sub"),
};
const ctx = els.canvas.getContext("2d");

function log(msg) { console.log(msg); els.log.textContent += msg + "\n"; }
function setStatus(msg, { spinner = true } = {}) {
  els.status.textContent = msg;
  els.overlay.hidden = false;
  els.overlay.querySelector(".spinner").style.visibility = spinner ? "visible" : "hidden";
}
function fail(msg) {
  setStatus(msg, { spinner: false });
  els.progress.hidden = true;
  els.next.disabled = true;
  log("ERROR: " + msg);
}

// ---------- downloads (Cache API + progress) ----------
async function fetchWithCache(url, sizeMB, onProgress) {
  let cache = null;
  try { cache = await caches.open("kattbilder-models"); } catch { /* unavailable */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { log(`${url.split("/").slice(-2).join("/")}: cached`); return new Uint8Array(await hit.arrayBuffer()); }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const declared = Number(res.headers.get("content-length")) || 0;
  const total = declared || sizeMB * 1024 * 1024;
  const reader = res.body.getReader();
  let buf = declared ? new Uint8Array(declared) : null;   // stream straight into one buffer when possible
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && received + value.byteLength > buf.byteLength) { chunks.push(buf.subarray(0, received)); buf = null; }
    if (buf) buf.set(value, received); else chunks.push(value);
    received += value.byteLength;
    onProgress(received, total);
  }
  if (!buf) {
    buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  } else if (received !== declared) {
    buf = buf.slice(0, received);
  }
  if (cache) {
    try { await cache.put(url, new Response(buf, { headers: { "content-length": String(received) } })); }
    catch (e) { log("cache.put failed (quota?): " + e.message); }
  }
  return buf;
}

async function download(name, url, sizeMB) {
  els.progress.hidden = false;
  els.bar.style.width = "0";
  setStatus(`Downloading ${name} (${sizeMB} MB, first time only)…`);
  const bytes = await fetchWithCache(url, sizeMB, (r, t) => {
    els.bar.style.width = `${Math.min(100, (100 * r) / t).toFixed(1)}%`;
    els.status.textContent = `Downloading ${name}… ${(r / 1048576).toFixed(0)} / ${(t / 1048576).toFixed(0)} MB`;
  });
  els.progress.hidden = true;
  return bytes;
}

async function loadSessions(set) {
  const ort = window.ort;
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;
  const baseOpt = {
    executionProviders: ["webgpu"],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: { session: {
      disable_prepacking: "1", use_device_allocator_for_initializers: "1",
      use_ort_model_bytes_directly: "1", use_ort_model_bytes_for_initializers: "1",
    } },
  };
  const resolve = (u) => (set.base.startsWith("http") ? `${set.base}/${u}` : new URL(`${set.base}/${u}`, location.href).href);
  const sessions = {};
  for (const name of Object.keys(set).filter((k) => k !== "base")) {
    const m = set[name];
    const t0 = performance.now();
    const bytes = await download(name, resolve(m.url), m.sizeMB);
    const opt = { ...baseOpt, ...m.opt };
    if (m.external) {
      opt.externalData = [];
      for (const [i, e] of m.external.entries()) {
        const ext = await download(`${name} weights ${i + 1}/${m.external.length}`, resolve(e.url), e.sizeMB);
        opt.externalData.push({ data: ext, path: e.url });
      }
    }
    setStatus(`Compiling ${name} for your GPU…`);
    sessions[name] = await ort.InferenceSession.create(bytes, opt);
    log(`${name}: ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  return sessions;
}

// ---------- math helpers ----------
function randn() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}
function randnArray(n, scale = 1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = randn() * scale;
  return a;
}
function toImageData(px, W, H) {   // [3,H,W] in [-1,1] -> RGBA
  const img = new ImageData(W, H);
  const plane = H * W;
  for (let i = 0; i < plane; i++) {
    img.data[i * 4] = clamp255(px[i]);
    img.data[i * 4 + 1] = clamp255(px[plane + i]);
    img.data[i * 4 + 2] = clamp255(px[2 * plane + i]);
    img.data[i * 4 + 3] = 255;
  }
  return img;
}
function clamp255(v) {
  const x = (v / 2 + 0.5) * 255;
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

// ---------- pipeline: SD-Turbo (1 step, fixed prompt) ----------
async function makeSdTurbo(precision) {
  const ort = window.ort;
  const [meta, bin] = await Promise.all([
    fetch("prompt_embeds.json").then((r) => r.json()),
    fetch("prompt_embeds.bin").then((r) => r.arrayBuffer()),
  ]);
  log(`prompt: "${meta.prompt}"`);
  const promptEmbeds = new ort.Tensor("float32", new Float32Array(bin), meta.dims);
  const sessions = await loadSessions(MODELS[`sdturbo_${precision}`]);
  const SIGMA = 14.6146, VAE_SCALE = 0.18215, SHAPE = [1, 4, 64, 64];
  const n = 4 * 64 * 64;

  return async function generate() {
    const latent = randnArray(n, SIGMA);
    const scaled = new Float32Array(n);
    const div = Math.sqrt(SIGMA * SIGMA + 1);
    for (let i = 0; i < n; i++) scaled[i] = latent[i] / div;

    const t0 = performance.now();
    const { out_sample } = await sessions.unet.run({
      sample: new ort.Tensor("float32", scaled, SHAPE),
      timestep: new ort.Tensor("int64", [999n], [1]),
      encoder_hidden_states: promptEmbeds,
    });
    const tUnet = performance.now() - t0;
    const eps = out_sample.data;
    const x0 = new Float32Array(n);
    for (let i = 0; i < n; i++) x0[i] = (latent[i] - SIGMA * eps[i]) / VAE_SCALE;
    out_sample.dispose?.();

    const t1 = performance.now();
    const { sample } = await sessions.vae_decoder.run({ latent_sample: new ort.Tensor("float32", x0, SHAPE) });
    const [, , H, W] = sample.dims;
    const img = toImageData(sample.data, W, H);
    sample.dispose?.();
    log(`cat: unet ${tUnet.toFixed(0)} ms, vae ${(performance.now() - t1).toFixed(0)} ms`);
    return createImageBitmap(img);
  };
}

// ---------- pipeline: DDPM cat 256 with DDIM sampling ----------
async function makeDdpm() {
  const ort = window.ort;
  const sessions = await loadSessions(MODELS.ddpm);
  const STEPS = Number(params.get("steps")) || 40;
  const ETA = params.get("eta") != null ? Number(params.get("eta")) : 0;
  const T = 1000, H = 256, W = 256, n = 3 * H * W, SHAPE = [1, 3, H, W];

  // linear beta schedule 1e-4 .. 0.02, alphas_cumprod
  const ac = new Float64Array(T);
  let prod = 1;
  for (let t = 0; t < T; t++) { prod *= 1 - (1e-4 + (0.02 - 1e-4) * t / (T - 1)); ac[t] = prod; }
  const stride = Math.floor(T / STEPS);
  const timesteps = [];
  for (let i = STEPS - 1; i >= 0; i--) timesteps.push(i * stride);   // "leading" spacing, as diffusers DDIM
  log(`ddpm: ${STEPS} DDIM steps, eta=${ETA}`);

  return async function generate() {
    let x = randnArray(n);
    const t0 = performance.now();
    for (let s = 0; s < timesteps.length; s++) {
      const t = timesteps[s];
      const { eps } = await sessions.unet.run({
        sample: new ort.Tensor("float32", x, SHAPE),
        timestep: new ort.Tensor("int64", [BigInt(t)], [1]),
      });
      const e = eps.data;
      const a = ac[t];
      const aPrev = s + 1 < timesteps.length ? ac[timesteps[s + 1]] : 1.0;
      const sqrtA = Math.sqrt(a), sqrt1mA = Math.sqrt(1 - a);
      const sigma = ETA * Math.sqrt((1 - aPrev) / (1 - a)) * Math.sqrt(1 - a / aPrev);
      const dirCoef = Math.sqrt(Math.max(0, 1 - aPrev - sigma * sigma));
      const sqrtAPrev = Math.sqrt(aPrev);
      const next = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let x0 = (x[i] - sqrt1mA * e[i]) / sqrtA;
        if (x0 > 1) x0 = 1; else if (x0 < -1) x0 = -1;            // clip_sample
        next[i] = sqrtAPrev * x0 + dirCoef * e[i] + (sigma > 0 ? sigma * randn() : 0);
      }
      eps.dispose?.();
      x = next;
    }
    log(`cat: ${STEPS} steps in ${(performance.now() - t0).toFixed(0)} ms`);
    return createImageBitmap(toImageData(x, W, H));
  };
}

// ---------- queue + UI ----------
const queue = [];
let producing = false;
const waiters = [];
function updateQueueLabel() { els.queue.textContent = `${queue.length} cat${queue.length === 1 ? "" : "s"} ready`; }

async function producer(generate) {
  if (producing) return;
  producing = true;
  try {
    while (queue.length < BATCH) {
      const bitmap = await generate();
      if (waiters.length) waiters.shift()(bitmap); else queue.push(bitmap);
      updateQueueLabel();
    }
  } catch (e) {
    fail("Generation failed: " + (e.message || e));
  } finally {
    producing = false;
  }
}
function takeNext(generate) {
  const p = queue.length ? Promise.resolve(queue.shift()) : new Promise((resolve) => waiters.push(resolve));
  updateQueueLabel();
  producer(generate);
  return p;
}
function show(bitmap) {
  els.canvas.width = bitmap.width;
  els.canvas.height = bitmap.height;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  els.canvas.hidden = false;
  els.overlay.hidden = true;
}

async function main() {
  if (!("gpu" in navigator)) return fail("WebGPU is not available in this browser. Use Chrome/Edge 113+ (or Firefox/Safari with WebGPU enabled).");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return fail("No WebGPU adapter found.");
  if (!window.ort) return fail("onnxruntime-web failed to load from the CDN.");
  const info = adapter.info || {};
  const desc = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ") || "unknown";
  const hasF16 = adapter.features.has("shader-f16");
  log(`WebGPU adapter: ${desc}; shader-f16: ${hasF16}`);
  if (info.architecture === "swiftshader") {
    return fail("WebGPU is running on SwiftShader (CPU emulation), not your GPU. See README for Chrome flags.");
  }

  const model = params.get("model") || (hasF16 ? "sdturbo" : "ddpm");
  const precision = params.get("precision") || "fp16";
  if (model === "sdturbo" && precision === "fp16" && !hasF16) {
    return fail(`Adapter "${desc}" lacks shader-f16, required by SD-Turbo fp16. Use ?model=ddpm instead.`);
  }
  els.sub.textContent = model === "sdturbo"
    ? `SD-Turbo (${precision}) on WebGPU, 512×512, fixed cat prompt. No server involved.`
    : "DDPM cat-256 (unconditional, trained on cat photos) on WebGPU, DDIM sampling. No server involved.";
  log(`pipeline: ${model}${model === "sdturbo" ? " " + precision : ""}`);

  let generate;
  try {
    generate = model === "sdturbo" ? await makeSdTurbo(precision) : await makeDdpm();
  } catch (e) {
    return fail("Loading models failed: " + (e.message || e));
  }

  setStatus("Generating the first cat…");
  els.next.addEventListener("click", async () => {
    els.next.disabled = true;
    if (!queue.length) setStatus("Generating…");
    show(await takeNext(generate));
    els.next.disabled = false;
  });
  show(await takeNext(generate));
  els.next.disabled = false;
}

main();
