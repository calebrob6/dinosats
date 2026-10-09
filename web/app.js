import * as ol from "./vendor/ol.js";
import { ClipTokenizer } from "./tokenizer.js";
import { PcaProjection, PositionalDebias, cosine, ensembleEmbedding, findPeaks, logSumExp, quantiles, relativeProbability, topStrings } from "./search.js";
import { createTour } from "./tour.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const params = new URLSearchParams(location.search);
const DEBUG = document.documentElement.classList.contains("debug");
// Minimum distance between listed top matches, independent of the embedding grid resolution.
const PEAK_SPACING_M = 75;
// The finest embedding level with at most this many tokens is loaded in full and scored for every query (scene-wide
// heatmap and top matches). Finer levels are streamed per COG block for the current view.
const MAX_GLOBAL_TOKENS = 2.5e6;
// A coarser level with at most this many tokens is loaded first, so the app is usable while the global one downloads.
const PREVIEW_TOKENS = 3e5;
// Stream the finest level whose tokens are at most this many screen pixels wide.
const LOD_PX = 3;
const MAX_BLOCKS = 48;
const MAX_FETCHES = 4;

const state = {
  config: null,
  scene: null,
  pca: null,
  projector: null, // PcaProjection
  extent: null, // [x0, y0, x1, y1] of the embedding COG
  levels: [], // embedding COG levels, finest first: {index, image, width, height, tile: [w, h], res: [x, y]}
  global: null, // region covering the whole scene at the finest level with at most MAX_GLOBAL_TOKENS tokens
  blocks: new Map(), // "level/bx/by" -> region; streamed blocks of levels finer than the global one
  grid: null, // {extent, width, height, res} of the global region
  globalFinal: false, // whether the global region is at its final level (not the preview)
  pool: null,
  projection: null,
  tokenizer: null,
  ort: null,
  session: null,
  backend: "",
  modelTag: "",
  embeddings: new Map(), // prompt -> Float32Array (patch half of the text embedding, ensembled over templates)
  weights: new Map(), // prompt -> {bias, w} (see PcaProjection.weights)
  background: { prompts: [], sig: "" },
  current: null, // {prompt, key, relative, lo, hi, cos, values, peaks} for the global region
  timing: {},
  vocab: null, // {strings, embeddings, index}: precomputed text embeddings (shift-click lookup and known prompts)
  point: null, // {x, y, region, index}: the shift-clicked token
  debias: null, // PositionalDebias while "Remove positional bias" is on, else null
  debiasParts: null, // {debias, norm}: the loaded positional basis and the offline per-token norm of the global level
  embeddingBytes: 0, // size of the embedding COG on disk
  facts: null, // scene numbers for the help tour (see describeScene)
};

// matplotlib inferno, 11 stops
const INFERNO = ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"];
const LUT = (() => {
  const stops = INFERNO.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const a = Math.min(stops.length - 2, Math.floor(x));
    const f = x - a;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[a][c] * (1 - f) + stops[a + 1][c] * f;
  }
  return lut;
})();

function setStatus(text) {
  $("status").textContent = text;
  $("status-row").dataset.state = /^Error/.test(text) ? "error" : /^(Ready|Showing)/.test(text) ? "ok" : "busy";
}

// Background work shown under the status line (full-resolution embeddings, text encoder download).
const activities = new Map();
function setActivity(key, text, fraction = null) {
  if (text == null) activities.delete(key);
  else activities.set(key, { text, fraction });
  const all = [...activities.values()];
  $("activity").hidden = !all.length;
  $("activity-text").textContent = all.map((a) => a.text).join(" · ");
  const bar = all.find((a) => a.fraction != null);
  $("progress").hidden = !bar;
  if (bar) $("progress").value = bar.fraction;
}

const scoreMode = () => document.querySelector('input[name="mode"]:checked').value;

/** Fills a range input's track up to its value (see input[type="range"] in style.css) and updates its output. */
function syncRange(input, digits = 2) {
  const f = (input.value - input.min) / (input.max - input.min);
  input.style.setProperty("--v", `${(100 * f).toFixed(1)}%`);
  const out = $(`${input.id}-out`);
  if (out) out.textContent = Number(input.value).toFixed(digits);
}

function fitPadding() {
  if (innerWidth > 820) return [24, 290, 24, 350];
  // Narrow screens: the query panel sits on top and the match list along the bottom.
  const top = $("left").getBoundingClientRect().bottom + 8;
  const bottom = innerHeight - $("right").getBoundingClientRect().top + 8;
  return [top, 8, Math.max(bottom, 36), 8];
}

async function fetchOk(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetchOk(url);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const buf = total ? new Uint8Array(total) : null;
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf) buf.set(value, loaded);
    else chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  if (buf) return buf;
  const out = new Uint8Array(loaded);
  chunks.reduce((off, c) => (out.set(c, off), off + c.length), 0);
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Map

function defineProjection(epsg) {
  const code = `EPSG:${epsg}`;
  if (ol.getProjection(code)) return code;
  let def;
  if (epsg >= 26901 && epsg <= 26923) def = `+proj=utm +zone=${epsg - 26900} +datum=NAD83 +units=m +no_defs`;
  else if (epsg >= 32601 && epsg <= 32660) def = `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs`;
  else if (epsg >= 32701 && epsg <= 32760) def = `+proj=utm +zone=${epsg - 32700} +south +datum=WGS84 +units=m +no_defs`;
  else throw new Error(`Unsupported CRS ${code}; add a proj4 definition in defineProjection()`);
  // Only UTM <-> lon/lat is needed (coordinate readout, scale line); ol/proj/proj4's register() would build
  // transforms between every pair of proj4 definitions.
  ol.addProjection(new ol.Projection({ code, units: "m" }));
  const t = ol.proj4("EPSG:4326", def);
  ol.addCoordinateTransforms(
    "EPSG:4326",
    code,
    (c) => t.forward(c),
    (c) => t.inverse(c),
  );
  return code;
}

let map;
let heatSource;
let heatLayer;
let markerSource;
let pointSource;
let naipLayer;

/** Shows or hides the heatmap layer (Space); the opacity readout says "off" while hidden. */
function setHeatVisible(visible) {
  if (!heatLayer) return;
  heatLayer.setVisible(visible);
  $("opacity").closest(".slider-row").classList.toggle("is-off", !visible);
  if (visible) syncRange($("opacity"));
  else $("opacity-out").textContent = "off";
}

function makeNaipSource(scene) {
  return new ol.GeoTIFFSource({
    sources: [{ url: scene.rgb, bands: [1, 2, 3], min: 0, max: 255 }],
    convertToRGB: "auto", // JPEG-compressed COGs are stored as YCbCr
    interpolate: true,
    transition: 0,
    attributions: "NAIP imagery: USDA FPAC, via Microsoft Planetary Computer",
  });
}

function buildMap(naipSource, projection, extent) {
  const naip = new ol.WebGLTileLayer({
    extent,
    source: naipSource,
    style: { color: ["array", ["band", 1], ["band", 2], ["band", 3], 1] },
  });
  naipLayer = naip;
  heatSource = new ol.ImageCanvasSource({ projection, ratio: 1, canvasFunction: renderHeat });
  heatLayer = new ol.ImageLayer({ source: heatSource, opacity: Number($("opacity").value) });
  markerSource = new ol.VectorSource();
  const markers = new ol.VectorLayer({
    source: markerSource,
    style: (f) =>
      new ol.Style({
        image: new ol.Circle({ radius: 10, fill: new ol.Fill({ color: "rgba(14,11,8,0.88)" }), stroke: new ol.Stroke({ color: "#e37148", width: 2 }) }),
        text: new ol.Text({ text: String(f.get("rank")), fill: new ol.Fill({ color: "#f3ecd8" }), font: '600 10px "JetBrains Mono", ui-monospace, monospace' }),
      }),
  });

  pointSource = new ol.VectorSource();
  const pointLayer = new ol.VectorLayer({
    source: pointSource,
    style: new ol.Style({
      image: new ol.Circle({ radius: 8, fill: new ol.Fill({ color: "rgba(229,168,83,0.2)" }), stroke: new ol.Stroke({ color: "#e5a853", width: 2.5 }) }),
    }),
  });

  map = new ol.Map({
    target: "map",
    layers: [naip, heatLayer, markers, pointLayer],
    controls: ol.defaultControls({ attributionOptions: { collapsible: true } }).extend([new ol.ScaleLine()]),
    view: new ol.View({ projection, enableRotation: false, maxResolution: 80, minResolution: 0.05, constrainResolution: false }),
  });
  map.getView().fit(extent, { padding: fitPadding() });

  $("naip").addEventListener("change", (e) => naip.setVisible(e.target.checked));
  $("opacity").addEventListener("input", (e) => {
    heatLayer.setOpacity(Number(e.target.value));
    setHeatVisible(true);
  });

  const hover = $("hover");
  map.on("singleclick", (e) => {
    if (e.originalEvent.shiftKey) selectPoint(e.coordinate);
  });
  map.on("pointermove", (e) => {
    const sample = !e.dragging && sampleAt(e.coordinate);
    if (!sample) {
      hover.hidden = true;
      return;
    }
    const value = `${state.current.relative ? "P" : "stretch"} ${sample.value.toFixed(3)}`;
    hover.textContent = DEBUG ? `${value} · cos ${sample.cos.toFixed(3)} · ${sample.res.toFixed(1)} m` : state.current.relative ? value : `cos ${sample.cos.toFixed(3)}`;
    hover.style.left = `${e.pixel[0] + 14}px`;
    hover.style.top = `${e.pixel[1] + 14}px`;
    hover.hidden = false;
  });
  map.getViewport().addEventListener("pointerleave", () => (hover.hidden = true));
}

function cellCenter(row, col) {
  const { extent, res } = state.grid;
  return [extent[0] + (col + 0.5) * res, extent[3] - (row + 0.5) * res];
}

const intersects = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];

/** The token under a map coordinate in the finest region accepted by `usable`, falling back to the global region. */
function tokenAt([x, y], usable = () => true) {
  if (!state.global) return null;
  const regions = [...state.blocks.values()].filter(usable).sort((a, b) => a.level - b.level);
  for (const region of [...regions, state.global]) {
    const { res } = state.levels[region.level];
    const col = Math.floor((x - state.extent[0]) / res[0]) - region.col0;
    const row = Math.floor((state.extent[3] - y) / res[1]) - region.row0;
    if (col < 0 || row < 0 || col >= region.width || row >= region.height) continue;
    return { region, col, row, index: row * region.width + col, res: res[0] };
  }
  return null;
}

/** Score at a map coordinate from the finest scored region under it. */
function sampleAt(xy) {
  const c = state.current;
  if (!c) return null;
  const t = tokenAt(xy, (b) => b.values?.key === c.key);
  return t && { value: t.region.values.data[t.index], cos: t.region.cos.data[t.index], res: t.res };
}

// ---------------------------------------------------------------------------------------------------------------
// Heatmap rendering: the global region everywhere, overdrawn by streamed blocks of the level that suits the zoom.

let redrawQueued = false;
function scheduleRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => {
    redrawQueued = false;
    heatSource?.changed();
  });
}

/** The level to stream for a view resolution (m per CSS px), or null if the global region is fine enough. */
function detailLevel(resolution) {
  let pick = 0;
  for (const l of state.levels) if (l.res[0] <= resolution * LOD_PX) pick = l.index;
  return pick < state.global.level ? state.levels[pick] : null;
}

function renderHeat(view, resolution, pixelRatio, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size[0];
  canvas.height = size[1];
  const level = state.current && detailLevel(resolution);
  requestBlocks(level ? visibleBlocks(level, view) : []);
  if (!state.current) return canvas;
  const t0 = performance.now();
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  const regions = [state.global];
  if (level) {
    // Cached blocks of coarser streamed levels stand in until the target level arrives.
    const detail = [...state.blocks.values()].filter((b) => b.level >= level.index && intersects(b.extent, view));
    detail.sort((a, b) => b.level - a.level);
    for (const b of detail) b.used = t0;
    regions.push(...detail);
  }
  for (const r of regions) drawRegion(ctx, r, view, size);
  state.timing.draw = performance.now() - t0;
  updateTiming();
  return canvas;
}

/** Draws the part of a region inside the view, snapped to whole canvas pixels so neighbouring blocks don't seam. */
function drawRegion(ctx, region, view, size) {
  const { res } = state.levels[region.level];
  const [ex0, , , ey1] = state.extent;
  const sx = size[0] / (view[2] - view[0]);
  const sy = size[1] / (view[3] - view[1]);
  const c0 = Math.max(region.col0, Math.floor((view[0] - ex0) / res[0]));
  const c1 = Math.min(region.col0 + region.width, Math.ceil((view[2] - ex0) / res[0]));
  const r0 = Math.max(region.row0, Math.floor((ey1 - view[3]) / res[1]));
  const r1 = Math.min(region.row0 + region.height, Math.ceil((ey1 - view[1]) / res[1]));
  if (c0 >= c1 || r0 >= r1) return;
  const left = Math.round((ex0 + c0 * res[0] - view[0]) * sx);
  const right = Math.round((ex0 + c1 * res[0] - view[0]) * sx);
  const top = Math.round((view[3] - ey1 + r0 * res[1]) * sy);
  const bottom = Math.round((view[3] - ey1 + r1 * res[1]) * sy);
  if (right <= left || bottom <= top) return;
  ctx.clearRect(left, top, right - left, bottom - top);
  ctx.drawImage(regionCanvas(region), c0 - region.col0, r0 - region.row0, c1 - c0, r1 - r0, left, top, right - left, bottom - top);
}

function regionCanvas(region) {
  const threshold = Number($("threshold").value);
  const key = `${state.current.key}\n${threshold}`;
  if (region.canvas?.key !== key) {
    const values = regionValues(region);
    const canvas = region.canvas?.el ?? document.createElement("canvas");
    canvas.width = region.width;
    canvas.height = region.height;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(region.width, region.height);
    const px = img.data;
    const span = Math.max(1e-6, 1 - threshold);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < threshold) continue;
      const c = Math.round(v * 255) * 3;
      px[i * 4] = LUT[c];
      px[i * 4 + 1] = LUT[c + 1];
      px[i * 4 + 2] = LUT[c + 2];
      px[i * 4 + 3] = 255 * Math.min(1, 0.35 + (0.65 * (v - threshold)) / span);
    }
    ctx.putImageData(img, 0, 0);
    region.canvas = { key, el: canvas };
  }
  return region.canvas.el;
}

// ---------------------------------------------------------------------------------------------------------------
// Embeddings: a COG level loaded in full up front, plus blocks of finer levels streamed for the view.

/** Reads the embedding COG's header: its levels and georeferencing, no pixel data. */
async function openEmbeddings(scene) {
  const tiff = await ol.fromUrl(scene.embeddings);
  const first = await tiff.getImage(0);
  const extent = first.getBoundingBox();
  const levels = [];
  let tileBytes = 0;
  for (let i = 0, n = await tiff.getImageCount(); i < n; i++) {
    const image = i ? await tiff.getImage(i) : first;
    tileBytes += [...(image.fileDirectory.TileByteCounts ?? [])].reduce((a, b) => a + b, 0);
    if ((image.fileDirectory.NewSubfileType ?? 0) & 4) continue; // transparency mask
    const width = image.getWidth();
    const height = image.getHeight();
    const res = [(extent[2] - extent[0]) / width, (extent[3] - extent[1]) / height];
    levels.push({ image, width, height, tile: [image.getTileWidth(), image.getTileHeight()], res });
  }
  levels.sort((a, b) => b.width - a.width).forEach((l, i) => (l.index = i));
  // Size on disk, from the Content-Range of the header requests.
  state.embeddingBytes = tiff.source?.fileSize || tileBytes;
  // A few decoder workers are plenty for the embedding tiles (the default is one per core, on top of OpenLayers' own pool).
  Object.assign(state, { extent, levels, pool: state.pool ?? new ol.Pool(Math.min(4, navigator.hardwareConcurrency || 2)) });
  return { epsg: first.geoKeys.ProjectedCSTypeGeoKey, bands: first.getSamplesPerPixel() };
}

async function loadLevel(level) {
  return level.image.readRasters({ interleave: true, pool: state.pool });
}

/** Makes a fully loaded level the global region and rescores the current query on it. */
async function setGlobal(level, codes, final) {
  const region = makeRegion("global", level, 0, 0, level.width, level.height, codes);
  while (state.debias && !region.inv) await ensureInv(region);
  state.global = region;
  state.globalFinal = final;
  state.grid = { extent: state.extent, width: level.width, height: level.height, res: level.res[0] };
  // Streamed blocks at or above the global level add nothing any more.
  for (const [key, b] of state.blocks) if (b.level >= level.index) state.blocks.delete(key);
  if (state.current) rescore();
  scheduleRedraw();
}

async function refineGlobal(level) {
  const bytes = [...(level.image.fileDirectory.TileByteCounts ?? [])].reduce((a, b) => a + b, 0);
  setActivity("refine", `Loading full-resolution embeddings${bytes ? ` · ${(bytes / 1e6).toFixed(0)} MB` : ""}`);
  try {
    await setGlobal(level, await loadLevel(level), true);
  } catch (err) {
    console.warn("could not load the full-resolution embeddings", err);
  } finally {
    setActivity("refine", null);
  }
}

function makeRegion(key, level, col0, row0, width, height, codes) {
  const [x0, , , y1] = state.extent;
  const [rx, ry] = level.res;
  const extent = [x0 + col0 * rx, y1 - (row0 + height) * ry, x0 + (col0 + width) * rx, y1 - row0 * ry];
  return { key, level: level.index, col0, row0, width, height, extent, codes, used: performance.now(), bg: null, cos: null, values: null, canvas: null };
}

/** Blocks of a level that intersect the view, nearest to the view centre first. */
function visibleBlocks(level, [vx0, vy0, vx1, vy1]) {
  const [ex0, , , ey1] = state.extent;
  const [bw, bh] = [level.tile[0] * level.res[0], level.tile[1] * level.res[1]];
  const nx = Math.ceil(level.width / level.tile[0]);
  const ny = Math.ceil(level.height / level.tile[1]);
  const bx0 = Math.max(0, Math.floor((vx0 - ex0) / bw));
  const bx1 = Math.min(nx - 1, Math.floor((vx1 - ex0) / bw));
  const by0 = Math.max(0, Math.floor((ey1 - vy1) / bh));
  const by1 = Math.min(ny - 1, Math.floor((ey1 - vy0) / bh));
  const cx = ((vx0 + vx1) / 2 - ex0) / bw - 0.5;
  const cy = (ey1 - (vy0 + vy1) / 2) / bh - 0.5;
  const out = [];
  for (let by = by0; by <= by1; by++) {
    for (let bx = bx0; bx <= bx1; bx++) out.push({ key: `${level.index}/${bx}/${by}`, level, bx, by, d: (bx - cx) ** 2 + (by - cy) ** 2 });
  }
  return out.sort((a, b) => a.d - b.d);
}

const fetches = { queue: [], inflight: new Set(), wanted: new Set() };

function requestBlocks(jobs) {
  fetches.wanted = new Set(jobs.map((j) => j.key));
  fetches.queue = jobs.filter((j) => !state.blocks.has(j.key) && !fetches.inflight.has(j.key));
  while (fetches.inflight.size < MAX_FETCHES && fetches.queue.length) loadBlock(fetches.queue.shift());
}

async function loadBlock({ key, level, bx, by }) {
  fetches.inflight.add(key);
  const stale = () => !!state.debias !== debiased;
  const debiased = !!state.debias;
  try {
    const col0 = bx * level.tile[0];
    const row0 = by * level.tile[1];
    const width = Math.min(level.tile[0], level.width - col0);
    const height = Math.min(level.tile[1], level.height - row0);
    const win = [col0, row0, col0 + width, row0 + height];
    const codes = await level.image.readRasters({ window: win, interleave: true, pool: state.pool });
    const region = makeRegion(key, level, col0, row0, width, height, codes);
    if (state.debias) region.inv = await state.debias.invNorm(codes);
    if (stale() || level.index >= state.global.level) return; // the debiasing or the global level changed meanwhile
    regionLse(region, state.current?.prompt); // background scores are the bulk of the work; do it before drawing
    state.blocks.set(key, region);
    if (state.blocks.size > MAX_BLOCKS) {
      const old = [...state.blocks.values()].filter((b) => !fetches.wanted.has(b.key)).sort((a, b) => a.used - b.used);
      for (const b of old.slice(0, state.blocks.size - MAX_BLOCKS)) state.blocks.delete(b.key);
    }
    if (fetches.wanted.has(key)) scheduleRedraw();
  } catch (err) {
    console.warn(`Could not load embedding block ${key}`, err);
  } finally {
    fetches.inflight.delete(key);
    while (fetches.inflight.size < MAX_FETCHES && fetches.queue.length) loadBlock(fetches.queue.shift());
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Text encoder

// Cache key: the content hash that `npm run build` writes next to the model, so re-uploading an identical model
// doesn't invalidate it. Falls back to size + last-modified if the hash file is missing.
async function modelVersion(url) {
  const res = await fetch(`${url}.sha256`, { cache: "no-cache" }).catch(() => null);
  const hash = res?.ok ? (await res.text()).trim().split(/\s+/)[0] : "";
  if (/^[0-9a-f]{64}$/.test(hash)) return hash;
  const head = await fetchOk(url, { method: "HEAD" });
  return `${head.headers.get("content-length")}-${head.headers.get("last-modified")}`;
}

async function loadModelBytes(url, tag, onProgress) {
  const key = `${new URL(url, location.href).href}?v=${encodeURIComponent(tag)}`;
  let cache = null;
  try {
    cache = await caches.open("dinotxt-models");
  } catch {
    // Cache API needs a secure context (localhost or https)
  }
  const hit = cache && (await cache.match(key));
  if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), tag, cached: true };
  const bytes = await fetchWithProgress(url, onProgress);
  if (cache) {
    for (const req of await cache.keys()) if (req.url.startsWith(key.split("?")[0])) await cache.delete(req);
    cache.put(key, new Response(bytes)).catch((err) => console.warn("could not cache model", err));
  }
  return { bytes, tag, cached: false };
}

function configureOrt(ort) {
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  ort.env.logLevel = "error";
}

async function runEncoder(texts) {
  const { ort, session, tokenizer } = state;
  const ids = texts.map((t) => tokenizer.tokenize(t));
  const len = Math.max(...ids.map((x) => x.length));
  // Zero padding after the end-of-text token is exact: the text transformer is causal and pools at the argmax token.
  const data = new BigInt64Array(ids.length * len);
  ids.forEach((row, i) => row.forEach((v, j) => (data[i * len + j] = BigInt(v))));
  const { text_embeds } = await session.run({ input_ids: new ort.Tensor("int64", data, [ids.length, len]) });
  return text_embeds.data;
}

async function selfCheck() {
  const ref = await fetchOk("selfcheck.json").then((r) => r.json());
  const ids = ref.ids;
  const { ort, session } = state;
  const { text_embeds } = await session.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]) });
  const e = text_embeds.data;
  let dot = 0;
  let norm = 0;
  for (let i = 0; i < e.length; i++) {
    dot += e[i] * ref.embedding[i];
    norm += e[i] * e[i];
  }
  return dot / Math.sqrt(norm);
}

// The exported model keeps its token embedding in fp16, so WebGPU is only usable with shader-f16.
async function webgpuSupported() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return !!adapter?.features.has("shader-f16");
  } catch {
    return false;
  }
}

async function initEncoder(model) {
  const want = params.get("ep") ?? ((await webgpuSupported()) ? "webgpu" : "wasm");
  if (want === "webgpu") {
    try {
      const ort = await import("./vendor/ort/ort.webgpu.min.mjs");
      configureOrt(ort);
      state.ort = ort;
      state.session = await ort.InferenceSession.create(model.bytes, {
        executionProviders: ["webgpu"],
        logSeverityLevel: 3,
      });
      const cos = await selfCheck();
      if (cos > 0.999) {
        state.backend = "WebGPU";
        return;
      }
      console.warn(`WebGPU text embedding differs from PyTorch (cos ${cos}); falling back to wasm`);
      await state.session.release();
    } catch (err) {
      console.warn("WebGPU backend unavailable, falling back to wasm", err);
    }
  }
  const ort = await import("./vendor/ort/ort.wasm.min.mjs");
  configureOrt(ort);
  // Run wasm inference in a worker so the map stays responsive while a prompt is being encoded.
  ort.env.wasm.proxy = params.get("proxy") !== "0";
  state.ort = ort;
  state.session = await ort.InferenceSession.create(model.bytes, { executionProviders: ["wasm"] });
  state.backend = `wasm × ${ort.env.wasm.numThreads}${ort.env.wasm.proxy ? " (worker)" : ""}`;
}

let sessionQueue = Promise.resolve();
function withSession(fn) {
  const result = sessionQueue.then(fn);
  sessionQueue = result.catch(() => {});
  return result;
}

// The text encoder (and its tokenizer and onnxruntime) is only downloaded when a prompt is not precomputed.
let encoderReady = null;
function ensureEncoder() {
  if (!encoderReady) {
    encoderReady = (async () => {
      const { config } = state;
      const tag = await modelTagReady;
      const progress = (loaded, total) =>
        setActivity("model", `Text encoder ${(loaded / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ""} MB`, total ? loaded / total : null);
      const [model, tokenizer] = await Promise.all([loadModelBytes(config.model, tag, progress), ClipTokenizer.load(config.vocab)]);
      state.tokenizer = tokenizer;
      setActivity("model", "Starting text encoder");
      await initEncoder(model);
      updateTiming();
    })().finally(() => setActivity("model", null));
    encoderReady.catch(() => (encoderReady = null)); // let the next prompt retry
  }
  return encoderReady;
}

let modelTagReady = Promise.resolve("");
let pcaReady = null;

function cacheKey(prompt) {
  return `dinotxt:${state.modelTag}:${state.pca.text.templates.join("|")}:${prompt}`;
}

// Same normalisation as the tokenizer, so a precomputed vector is exactly what the encoder would see.
const normalizePrompt = (s) => s.replace(/\s+/gu, " ").trim().toLowerCase();

/**
 * The text embedding of a prompt: precomputed (vocab), cached from an earlier visit (localStorage), or encoded in the
 * browser. Calls `onEncode` before downloading or running the encoder, and returns null if `stale()` turns true first.
 */
async function embedPrompt(prompt, { stale = () => false, onEncode = () => {} } = {}) {
  if (state.embeddings.has(prompt)) return state.embeddings.get(prompt);
  const known = (await vocabReady)?.index.get(normalizePrompt(prompt));
  if (known) {
    state.embeddings.set(prompt, known);
    return known;
  }
  await Promise.all([modelTagReady, pcaReady]);
  try {
    const stored = localStorage.getItem(cacheKey(prompt));
    if (stored) {
      const bytes = Uint8Array.from(atob(stored), (c) => c.charCodeAt(0));
      const t = new Float32Array(bytes.buffer);
      state.embeddings.set(prompt, t);
      return t;
    }
  } catch {
    // ignore storage errors
  }
  if (stale()) return null;
  onEncode();
  await ensureEncoder();
  const { templates, embedding_slice: slice } = state.pca.text;
  const t0 = performance.now();
  const data = await withSession(() => (stale() ? null : runEncoder(templates.map((s) => s.replace("{}", prompt)))));
  if (!data) return null;
  state.timing.text = performance.now() - t0;
  const t = ensembleEmbedding(data, templates.length, slice);
  state.embeddings.set(prompt, t);
  try {
    localStorage.setItem(cacheKey(prompt), btoa(String.fromCharCode(...new Uint8Array(t.buffer))));
  } catch {
    // quota exceeded: keep the in-memory copy only
  }
  return t;
}

// ---------------------------------------------------------------------------------------------------------------
// Background prompts and scoring

function parseLines(text) {
  const seen = new Set();
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s && !seen.has(s) && seen.add(s));
}

let backgroundReady = Promise.resolve();
function updateBackground() {
  const prompts = parseLines($("background").value);
  backgroundReady = (async () => {
    for (const [i, p] of prompts.entries()) {
      await embedPrompt(p, { onEncode: () => setStatus(`Encoding background prompts ${i + 1}/${prompts.length}…`) });
    }
    state.background = { prompts, sig: prompts.join("\n") };
    if (state.global) regionLse(state.global, state.current?.prompt);
  })();
  return backgroundReady;
}

function promptWeights(prompt) {
  if (!state.weights.has(prompt)) {
    const t = state.embeddings.get(prompt);
    state.weights.set(prompt, state.projector.weights(state.debias ? state.debias.projectOut(t) : t));
  }
  return state.weights.get(prompt);
}

/** cos(t, x) of every token of a region, or cos(t, x') after positional debiasing (see PositionalDebias). */
function scoreRegion(region, prompt) {
  const out = cosine(region.codes, promptWeights(prompt));
  if (state.debias) for (let i = 0; i < out.length; i++) out[i] *= region.inv[i];
  return out;
}

function regionCos(region, prompt) {
  if (region.cos?.prompt !== prompt) region.cos = { prompt, data: scoreRegion(region, prompt) };
  return region.cos.data;
}

/**
 * Per-token log-sum-exp of the scaled background scores, leaving out the query if it is itself a background prompt.
 * Streamed blocks keep only the log-sum-exp; the global region also keeps the per-prompt scores for quick exclusion.
 */
function regionLse(region, prompt) {
  const { prompts, sig } = state.background;
  if (region.bg?.sig !== sig) region.bg = { sig, lse: new Map(), scores: region === state.global ? new Map() : null };
  const exclude = prompts.indexOf(prompt);
  if (!region.bg.lse.has(exclude)) {
    const scores = prompts
      .filter((_, i) => i !== exclude)
      .map((p) => {
        let s = region.bg.scores?.get(p);
        if (!s) {
          s = scoreRegion(region, p);
          region.bg.scores?.set(p, s);
        }
        return s;
      });
    region.bg.lse.set(exclude, scores.length ? logSumExp(scores, state.pca.text.logit_scale) : null);
  }
  return region.bg.lse.get(exclude);
}

/** Heatmap values of a region for the current query, mode and background (cached on the region). */
function regionValues(region) {
  const c = state.current;
  if (region.values?.key !== c.key) {
    const cos = regionCos(region, c.prompt);
    let data;
    if (c.relative) {
      data = relativeProbability(cos, regionLse(region, c.prompt), state.pca.text.logit_scale);
    } else {
      data = new Float32Array(cos.length);
      for (let i = 0; i < cos.length; i++) data[i] = Math.min(1, Math.max(0, (cos[i] - c.lo) / (c.hi - c.lo || 1)));
    }
    region.values = { key: c.key, data };
  }
  return region.values.data;
}

function drawLegend() {
  const ctx = $("legend-bar").getContext("2d");
  const img = ctx.createImageData(256, 1);
  for (let i = 0; i < 256; i++) img.data.set([LUT[i * 3], LUT[i * 3 + 1], LUT[i * 3 + 2], 255], i * 4);
  ctx.putImageData(img, 0, 0);
}

function updateLegend() {
  const c = state.current;
  const relative = c ? c.relative : scoreMode() === "relative" && state.background.prompts.length > 0;
  $("legend-title").textContent = relative ? "P(query vs background)" : "cosine";
  $("legend-lo").textContent = relative || !c ? "0" : c.lo.toFixed(3);
  $("legend-hi").textContent = relative || !c ? "1" : c.hi.toFixed(3);
}

function updateMatches() {
  const list = $("matches");
  markerSource?.clear();
  if (!state.current || !markerSource) {
    list.replaceChildren();
    return;
  }
  if (!$("peaks-on").checked) {
    state.current.peaks = [];
    list.replaceChildren();
    return;
  }
  const min = Math.max(0.5, Number($("threshold").value));
  const radius = Math.max(1, Math.round(PEAK_SPACING_M / state.grid.res));
  const peaks = findPeaks(state.current.values, state.grid.width, { count: 10, radius, min });
  state.current.peaks = peaks;
  if (!peaks.length) {
    list.replaceChildren(el("li", { className: "empty", textContent: `Nothing above ${min.toFixed(2)}` }));
    return;
  }
  list.replaceChildren(
    ...peaks.map((p, i) => {
      const center = cellCenter(p.row, p.col);
      markerSource.addFeature(new ol.Feature({ geometry: new ol.Point(center), rank: i + 1 }));
      const [lon, lat] = ol.transform(center, state.projection, "EPSG:4326");
      const link = el("a", { className: "loc" }, [
        el("span", { className: "loc-num", textContent: String(i + 1) }),
        el("span", { className: "loc-body" }, [
          el("span", { className: "loc-name", textContent: `${state.current.relative ? "P " : ""}${p.value.toFixed(3)}` }),
          el("span", { className: "loc-desc", textContent: `${lat.toFixed(5)}, ${lon.toFixed(5)}` }),
        ]),
      ]);
      link.addEventListener("click", () => map.getView().animate({ center, resolution: 0.8, duration: 450 }));
      return el("li", {}, [link]);
    }),
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Positional debiasing (debug only)

const loadFloats = (url) => fetchOk(url).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b));

/** Drops everything derived from the embeddings' scores (not the embeddings themselves). */
function resetScores() {
  state.weights.clear();
  state.blocks.clear();
  fetches.queue = [];
  fetches.wanted = new Set();
  Object.assign(state.global, { bg: null, cos: null, values: null, canvas: null });
}

/** Per-token norm correction of a region after debiasing: from the offline `posnorm` file when it fits, else computed. */
async function ensureInv(region) {
  if (!state.debias || region.inv) return;
  const { norm } = state.debiasParts;
  region.inv = norm?.length === region.width * region.height ? norm : await state.debias.invNorm(region.codes, (f) => setStatus(`Computing positional correction ${Math.round(100 * f)}%…`));
}

// Enables or disables positional debiasing; streamed blocks compute their own correction as they load.
async function applyDebias() {
  const want = $("debias").checked && !!state.scene.posbasis;
  if (want && !state.debiasParts) {
    setStatus("Loading positional basis…");
    const [basis, norm] = await Promise.all([loadFloats(state.scene.posbasis), state.scene.posnorm ? loadFloats(state.scene.posnorm).catch(() => null) : null]);
    state.debiasParts = { debias: new PositionalDebias(basis, state.projector), norm };
  }
  state.debias = want ? state.debiasParts.debias : null;
  while (state.debias && !state.global.inv) await ensureInv(state.global); // the global region may be replaced meanwhile
  resetScores();
}

function formatCount(n) {
  const [v, unit] = n >= 1e9 ? [n / 1e9, "B"] : n >= 1e6 ? [n / 1e6, "M"] : n >= 1e3 ? [n / 1e3, "K"] : [n, ""];
  return [v >= 100 ? v.toFixed(0) : v.toPrecision(3), unit];
}

function describeScene() {
  const { scene, levels, pca, extent } = state;
  const fine = levels[0];
  const tokenPx = pca.token_size_px;
  const gsd = fine.res[0] / tokenPx;
  const [w, h] = [Math.round(fine.width * tokenPx), Math.round(fine.height * tokenPx)];
  const km2 = ((extent[2] - extent[0]) * (extent[3] - extent[1])) / 1e6;
  const tokens = fine.width * fine.height;
  state.facts = { w, h, gsd, km2, tokens, tokenRes: fine.res[0], bytes: state.embeddingBytes, rawBytes: tokens * pca.input_dim * 4 };
  $("scene-name").textContent = scene.name;
  $("scene-sub").textContent = [scene.subtitle, `${+gsd.toFixed(2)} m`].filter(Boolean).join(" · ");
  const stat = (value, label, title) => {
    const [v, unit] = value;
    return el("div", { className: "stat", title }, [el("div", { className: "stat-value" }, [v, el("small", { textContent: unit })]), el("div", { className: "stat-label", textContent: label })]);
  };
  $("scene-stats").replaceChildren(
    stat(formatCount(w * h), "Pixels", `${w.toLocaleString()} × ${h.toLocaleString()} = ${(w * h).toLocaleString()} pixels searched`),
    stat([km2 >= 100 ? km2.toFixed(0) : km2.toPrecision(3), "km²"], "Area", `${km2.toFixed(1)} km²`),
    stat(formatCount(tokens), "Embeddings", `${tokens.toLocaleString()} DINOv3 patch embeddings, one per ${fine.res[0].toFixed(1)} m`),
  );
  $("debias").disabled = !scene.posbasis;
}

const formatBytes = (n) => (n >= 1e9 ? `${(n / 1e9).toPrecision(3)} GB` : `${(n / 1e6).toFixed(0)} MB`);
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

// ---------------------------------------------------------------------------------------------------------------
// Help tour: the "?" in the title card; it also opens on load unless "Don't show this on startup" was checked.

function setupHelp() {
  const { config, scene, facts: f, pca } = state;
  const words = { B: " billion", M: " million", K: " thousand", "": "" };
  const count = (n) => {
    const [v, unit] = formatCount(n);
    return v + words[unit];
  };
  const km2 = f.km2 >= 100 ? f.km2.toFixed(0) : f.km2.toPrecision(3);
  const res = `${+f.tokenRes.toFixed(1)} m`;
  const imagery = [scene.subtitle, "imagery"].filter(Boolean).join(" ");
  const bg = config.background.slice(0, 3).map((p) => `<em>${escapeHtml(p)}</em>`);
  const bgList = bg.length > 1 ? `${bg.slice(0, -1).join(", ")} and ${bg.at(-1)}` : bg.join("");
  let encoderBytes = 0;
  fetch(config.model, { method: "HEAD" })
    .then((r) => (encoderBytes = Number(r.headers.get("content-length")) || 0))
    .catch(() => {});
  const steps = [
    {
      label: "Welcome",
      title: "Search aerial imagery with words",
      body: () =>
        `<p>Describe what you're looking for and DinoSats maps where it is across <strong>${km2} km²</strong> of ${escapeHtml(imagery)} over ${escapeHtml(scene.name)}: <strong>${count(f.w * f.h)} pixels</strong> (${f.w.toLocaleString()} × ${f.h.toLocaleString()} at ${+f.gsd.toFixed(2)} m).</p>` +
        `<p>It all runs in your browser, over <strong>${formatBytes(f.bytes)}</strong> of precomputed image embeddings.</p>` +
        `<p>Use <kbd>→</kbd> or <kbd>Enter</kbd> to step through and <kbd>Esc</kbd> to close.</p>`,
    },
    {
      label: "Step 1 / 4",
      title: "Describe what to find",
      target: () => $("query-card"),
      body: () =>
        `<p>Type a phrase or pick an example. <strong>${state.vocab ? state.vocab.strings.length.toLocaleString() : "Hundreds of"} common phrases</strong> have precomputed text embeddings and show up instantly.</p>` +
        `<p>Anything else first downloads the ${encoderBytes ? `${formatBytes(encoderBytes)} ` : ""}dino.txt text encoder, once; the browser keeps it for later visits.</p>` +
        `<p><kbd>/</kbd> jumps to the search box.</p>`,
    },
    {
      label: "Step 2 / 4",
      title: "Read the heatmap",
      target: () => $("display"),
      body: () =>
        `<p><strong>Relative</strong> scores each ${res} patch by how much more it looks like your query than like ${config.background.length} background prompts such as ${bgList}: a probability from 0 to 1. <strong>Raw cosine</strong> shows the plain text–image similarity.</p>` +
        `<p>Searching for a background prompt itself switches to raw cosine for that query. <strong>Threshold</strong> hides weak scores, and <kbd>Space</kbd> shows or hides the heatmap.</p>`,
    },
    {
      label: "Step 3 / 4",
      title: "Explore the matches",
      target: () => $("matches-card"),
      placement: "left",
      body: () =>
        `<p>Switch on <strong>Top matches</strong> to list the best spots, and click one to fly there. Zooming in streams finer embeddings for the view, down to one every ${res}.</p>` +
        `<p><kbd>Shift</kbd> + click anywhere to see which phrases describe that spot best.</p>`,
    },
    {
      label: "Step 4 / 4",
      title: "How it works",
      body: () =>
        `<p>Offline, a <strong>DINOv3 ViT-L/16</strong> backbone embedded the imagery at 2× upsampling: one ${pca.input_dim.toLocaleString()}-d embedding per ${res} patch, <strong>${count(f.tokens)}</strong> in all. PCA squeezes each into ${pca.n_components} bytes, so the whole scene is a <strong>${formatBytes(f.bytes)}</strong> Cloud-Optimized GeoTIFF instead of ${formatBytes(f.rawBytes)} of float32.</p>` +
        `<p>The browser fetches only the parts it needs with HTTP range requests and scores every patch against the <strong>dino.txt</strong> embedding of your query.</p>` +
        `<p>Reopen this guide any time with <strong>?</strong>.</p>`,
    },
  ];
  const tour = createTour({ steps, trigger: $("help"), storageKey: "dinosats:hide-help" });
  $("help").disabled = false;
  tour.autoStart();
}

// ---------------------------------------------------------------------------------------------------------------
// Shift-click reverse lookup: the strings whose text embeddings best match one token's patch embedding.

function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    const v = e === 0 ? f * 2 ** -24 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15);
    out[i] = h & 0x8000 ? -v : v;
  }
  return out;
}

let vocabReady = Promise.resolve(null);
async function loadVocab(config) {
  if (!config.point_vocab) return null;
  const [meta, buf] = await Promise.all([
    fetchOk(config.point_vocab.strings).then((r) => r.json()),
    fetchOk(config.point_vocab.embeddings).then((r) => r.arrayBuffer()),
  ]);
  const all = meta.dtype === "float16" ? halfToFloat(new Uint16Array(buf)) : new Float32Array(buf);
  if (all.length !== meta.strings.length * meta.dim) throw new Error("vocab embeddings do not match vocab strings");
  const embeddings = meta.strings.map((_, i) => all.subarray(i * meta.dim, (i + 1) * meta.dim));
  const index = new Map(meta.strings.map((s, i) => [normalizePrompt(s), embeddings[i]]));
  return { strings: meta.strings, embeddings, index };
}

async function selectPoint(xy) {
  const t = tokenAt(xy);
  state.point = t && { ...t, xy };
  await vocabReady;
  renderPoint();
  if (!state.point || !state.vocab) return;
  // Show the clicked point's phrases and minimize the top matches (a new query brings them back).
  setCollapsed($("point-card"), false);
  if ($("peaks-on").checked && !$("matches-card").classList.contains("is-collapsed")) {
    setCollapsed($("matches-card"), true);
    matchesAutoCollapsed = true;
  }
}

// The Top matches and Clicked point cards minimize to their header (click the title). Only one is expanded
// automatically: a shift-click shows the point's phrases, a new query shows the top matches.
let matchesAutoCollapsed = false;
function setCollapsed(card, collapsed) {
  card.classList.toggle("is-collapsed", collapsed);
  card.querySelector(".card-toggle").setAttribute("aria-expanded", String(!collapsed));
  if (card.id === "matches-card") matchesAutoCollapsed = false;
}

function showMatches() {
  if (!$("peaks-on").checked) return;
  setCollapsed($("matches-card"), false);
  if (!$("point-card").hidden) setCollapsed($("point-card"), true);
}

function renderPoint() {
  const info = $("point-info");
  const list = $("point-matches");
  pointSource?.clear();
  $("point-card").hidden = !state.point || !state.vocab;
  if (!state.point || !state.vocab) {
    list.replaceChildren();
    return;
  }
  const { region, index, res } = state.point;
  const center = [
    state.extent[0] + (region.col0 + (index % region.width) + 0.5) * res,
    state.extent[3] - (region.row0 + Math.floor(index / region.width) + 0.5) * res,
  ];
  pointSource.addFeature(new ol.Feature({ geometry: new ol.Point(center) }));
  const [lon, lat] = ol.transform(center, state.projection, "EPSG:4326");
  let x = state.projector.pixelEmbedding(region.codes, index);
  if (state.debias) x = state.debias.projectOut(x);
  // Candidates: the precomputed vocabulary plus every other prompt encoded this session (queries and background).
  const extra = [...state.embeddings].filter(([p]) => !state.vocab.index.has(normalizePrompt(p)));
  const typed = { strings: extra.map(([p]) => p), embeddings: extra.map(([, t]) => t) };
  const top = topStrings(x, [state.vocab, typed], state.pca.text.logit_scale, 10);
  info.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  info.title = `${res.toFixed(1)} m token; cosine, then softmax share over ${state.vocab.strings.length + typed.strings.length} strings`;
  list.replaceChildren(
    ...top.map((m) => {
      const link = el("a", { textContent: m.text, title: "Search for this" });
      link.addEventListener("click", () => setQuery(m.text));
      return el("li", {}, [link, el("span", { className: "score", textContent: `${m.cos.toFixed(3)} · ${(100 * m.prob).toFixed(0)}%` })]);
    }),
  );
}

function updateTiming() {
  const t = state.timing;
  const parts = [];
  if (t.text != null) parts.push(`text ${Math.round(t.text)} ms`);
  if (t.score != null) parts.push(`score ${Math.round(t.score)} ms`);
  if (t.draw != null) parts.push(`draw ${Math.round(t.draw)} ms`);
  if (state.grid) parts.push(`global ${state.grid.res.toFixed(1)} m`);
  parts.push(state.backend || "encoder not loaded");
  $("timing").textContent = parts.join(" · ");
}

/** Scores the global region for the current query; streamed blocks are scored lazily when drawn. */
function rescore() {
  const c = state.current;
  if (!c) return;
  const t0 = performance.now();
  const g = state.global;
  const cos = regionCos(g, c.prompt);
  const lse = scoreMode() === "relative" ? regionLse(g, c.prompt) : null;
  c.relative = lse != null;
  if (!c.relative) g.cos.stretch ??= quantiles(cos, [0.5, 0.999]);
  [c.lo, c.hi] = c.relative ? [0, 1] : g.cos.stretch;
  c.key = [c.prompt, c.relative, state.background.sig, c.lo, c.hi].join("\n");
  c.cos = cos;
  c.values = regionValues(g);
  state.timing.score = performance.now() - t0;
  updateMatches();
  updateLegend();
  updateTiming();
  scheduleRedraw();
}

let startupReady = null; // resolves once the map, a global level and the background prompts are ready
let queryToken = 0;

// A background prompt is mostly scored against near-synonyms of itself, so searching for one switches to raw cosine
// until the next query that isn't one (unless the mode is changed by hand in between).
let autoRaw = false;
function syncAutoMode(prompt) {
  const key = prompt ? normalizePrompt(prompt) : "";
  const isBackground = !!key && state.background.prompts.some((p) => normalizePrompt(p) === key);
  if (isBackground && scoreMode() === "relative") {
    $("mode-raw").checked = true;
    autoRaw = true;
  } else if (!isBackground && autoRaw) {
    $("mode-relative").checked = true;
    autoRaw = false;
  }
  const note = $("mode-note");
  note.hidden = !(isBackground && autoRaw);
  if (!note.hidden) note.replaceChildren("“", el("strong", { textContent: prompt }), "” is a background prompt, so this shows raw cosine similarity. Other queries switch back to relative.");
}

/** Shows a query; a newer call makes older ones stop as soon as they notice (they never touch the display). */
async function requestQuery(prompt, { typed = false } = {}) {
  const token = ++queryToken;
  const stale = () => token !== queryToken;
  for (const b of $("examples").children) b.classList.toggle("active", b.textContent === prompt);
  try {
    if (prompt) {
      const short = prompt.length > 40 ? prompt.slice(0, 40) + "…" : prompt;
      const t = await embedPrompt(prompt, {
        stale,
        onEncode: () => {
          setStatus(state.session ? `Encoding “${short}”…` : `Loading the text encoder for “${short}”…`);
          state.timing.text = null;
        },
      });
      if (!t || stale()) return;
    }
    await startupReady;
    await backgroundReady;
    if (stale()) return;
    if (!prompt) {
      syncAutoMode("");
      state.current = null;
      scheduleRedraw();
      updateMatches();
      updateLegend();
      setStatus("Ready");
      return;
    }
    syncAutoMode(prompt);
    setHeatVisible(true);
    state.current = { prompt };
    rescore();
    setStatus(`Showing “${prompt}”`);
    if (typed) showMatches();
  } catch (err) {
    if (stale()) return;
    console.error(err);
    setStatus(`Error: ${err.message}`);
  }
}

function setQuery(text) {
  $("query").value = text;
  history.replaceState(null, "", text ? `#q=${encodeURIComponent(text)}` : location.pathname + location.search);
  requestQuery(text.trim(), { typed: true });
}

// ---------------------------------------------------------------------------------------------------------------

async function main() {
  // ?config=<url> loads another config (resolved against the page, as are the asset paths inside it).
  const config = await fetchOk(params.get("config") || "config.json").then((r) => r.json());
  const scene = config.scene;
  Object.assign(state, { config, scene });
  // Start everything that doesn't depend on the embedding header right away.
  const naipSource = makeNaipSource(scene);
  pcaReady = fetchOk(scene.pca)
    .then((r) => r.json())
    .then((pca) => Object.assign(state, { pca, projector: new PcaProjection(pca) }).pca);
  vocabReady = loadVocab(config)
    .catch((err) => (console.warn("no precomputed vocab", err), null))
    .then((vocab) => (state.vocab = vocab));
  modelTagReady = modelVersion(config.model)
    .catch(() => "")
    .then((tag) => (state.modelTag = tag));
  let done;
  startupReady = new Promise((resolve) => (done = resolve));

  $("background").value = config.background.join("\n");
  $("examples").replaceChildren(
    ...config.examples.map((x) => {
      const b = el("button", { type: "button", textContent: x });
      b.addEventListener("click", () => setQuery(x));
      return b;
    }),
  );
  drawLegend();
  updateLegend();
  for (const id of ["threshold", "opacity"]) syncRange($(id));
  let timer;
  $("query").addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => setQuery(e.target.value), 300);
  });
  $("query-form").addEventListener("submit", (e) => {
    e.preventDefault();
    clearTimeout(timer);
    setQuery($("query").value);
    $("query").blur(); // so Space toggles the heatmap right away (and the on-screen keyboard closes)
  });
  // "/" focuses the search box and Space toggles the heatmap, unless typing (or in the help tour, which has its own keys).
  const ignoreKey = (e) =>
    e.ctrlKey || e.metaKey || e.altKey || e.target.isContentEditable || e.target.closest?.("textarea, select, .tut-card button, .tut-card input, input:not([type=checkbox], [type=radio], [type=range])");
  document.addEventListener("keydown", (e) => {
    if (ignoreKey(e)) return;
    if (e.key === "/") {
      e.preventDefault();
      $("query").focus();
      $("query").select();
    } else if (e.key === " ") {
      e.preventDefault(); // also keeps a focused chip or checkbox from activating
      if (!e.repeat) setHeatVisible(!heatLayer?.getVisible());
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.key === " " && !ignoreKey(e)) e.preventDefault();
  });
  const initial = new URLSearchParams(location.hash.slice(1)).get("q");
  if (initial) setQuery(initial);
  $("query").focus();

  const header = await openEmbeddings(scene);
  // A coarse level first so the heatmap is usable quickly; the scene-wide level follows in the background.
  const maxTokens = scene.max_global_tokens ?? MAX_GLOBAL_TOKENS;
  const target = state.levels.find((l) => l.width * l.height <= maxTokens) ?? state.levels.at(-1);
  const preview = state.levels.find((l) => l.width * l.height <= PREVIEW_TOKENS && l.index > target.index) ?? target;
  const previewCodes = loadLevel(preview);
  setStatus("Loading embeddings…");
  state.projection = defineProjection(header.epsg);
  buildMap(naipSource, state.projection, state.extent);

  const pca = await pcaReady;
  if (header.bands !== pca.n_components) throw new Error(`${scene.embeddings} has ${header.bands} bands, PCA model has ${pca.n_components}`);
  describeScene();
  setupHelp();
  await setGlobal(preview, await previewCodes, preview === target);
  if (params.get("debias") === "1") {
    $("debias").checked = true;
    await applyDebias();
  }
  await updateBackground();
  updateLegend();
  setStatus("Ready");
  updateTiming();
  done();
  if (preview !== target) refineGlobal(target);

  for (const id of ["mode-relative", "mode-raw"]) {
    $(id).addEventListener("change", () => {
      autoRaw = false;
      $("mode-note").hidden = true;
      rescore();
      updateLegend();
    });
  }
  $("matches-card").querySelector(".card-toggle").disabled = !$("peaks-on").checked; // the browser may restore the switch
  $("peaks-on").addEventListener("change", (e) => {
    $("matches-card").querySelector(".card-toggle").disabled = !e.target.checked;
    setCollapsed($("matches-card"), false);
    updateMatches();
  });
  for (const id of ["matches-card", "point-card"]) {
    const card = $(id);
    card.querySelector(".card-toggle").addEventListener("click", () => setCollapsed(card, !card.classList.contains("is-collapsed")));
  }
  $("point-close").addEventListener("click", () => {
    if (matchesAutoCollapsed) setCollapsed($("matches-card"), false);
    state.point = null;
    renderPoint();
  });
  $("debias").addEventListener("change", async () => {
    $("debias").disabled = true;
    const prompt = state.current?.prompt;
    try {
      state.current = null;
      updateMatches();
      await applyDebias();
      await updateBackground();
      renderPoint();
      if (prompt) await requestQuery(prompt);
      else setStatus("Ready");
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    } finally {
      $("debias").disabled = !state.scene.posbasis;
    }
  });
  $("threshold").addEventListener("input", () => {
    syncRange($("threshold"));
    rescore();
  });
  let bgTimer;
  $("background").addEventListener("input", () => {
    clearTimeout(bgTimer);
    bgTimer = setTimeout(async () => {
      await updateBackground();
      syncAutoMode(state.current?.prompt);
      rescore();
      setStatus(state.current ? `Showing “${state.current.prompt}”` : "Ready");
    }, 600);
  });
}

main().catch((err) => {
  console.error(err);
  setStatus(`Error: ${err.message}`);
});

// Exposed for debugging and automated checks.
window.__app = {
  state,
  setQuery,
  fetches,
  get heatVisible() {
    return heatLayer?.getVisible();
  },
};
