// Bundles OpenLayers + proj4 + geotiff into web/vendor/ol.js and copies onnxruntime-web and coi-serviceworker.
// Run: npm run build
import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const web = join(root, "web");
const vendor = join(web, "vendor");
await mkdir(join(vendor, "ort"), { recursive: true });

await build({
  stdin: {
    contents: `
      export { default as Map } from "ol/Map.js";
      export { default as View } from "ol/View.js";
      export { default as Feature } from "ol/Feature.js";
      export { default as Point } from "ol/geom/Point.js";
      export { default as WebGLTileLayer } from "ol/layer/WebGLTile.js";
      export { default as ImageLayer } from "ol/layer/Image.js";
      export { default as VectorLayer } from "ol/layer/Vector.js";
      export { default as GeoTIFFSource } from "ol/source/GeoTIFF.js";
      export { default as ImageCanvasSource } from "ol/source/ImageCanvas.js";
      export { default as VectorSource } from "ol/source/Vector.js";
      export { Style, Circle, Fill, Stroke, Text } from "ol/style.js";
      export { defaults as defaultControls, ScaleLine } from "ol/control.js";
      export { default as Projection } from "ol/proj/Projection.js";
      export { addCoordinateTransforms, addProjection, transform, get as getProjection } from "ol/proj.js";
      export { default as proj4 } from "proj4";
      export { fromUrl, Pool } from "geotiff";
    `,
    resolveDir: root,
    loader: "js",
  },
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  sourcemap: true,
  outfile: join(vendor, "ol.js"),
  logLevel: "warning",
});
await copyFile(join(root, "node_modules/ol/ol.css"), join(vendor, "ol.css"));

// The wasm build is used when WebGPU is unavailable; the WebGPU build also needs its own (asyncify) wasm.
const ortDist = join(root, "node_modules/onnxruntime-web/dist");
for (const f of [
  "ort.wasm.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort.webgpu.min.mjs",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
]) {
  await copyFile(join(ortDist, f), join(vendor, "ort", f));
}

// Adds COOP/COEP headers client side on static hosts that can't set them (needed for multi-threaded wasm).
await copyFile(join(root, "node_modules/coi-serviceworker/coi-serviceworker.min.js"), join(web, "coi-serviceworker.js"));

console.log("built web/vendor/ol.js, web/vendor/ort/ and web/coi-serviceworker.js");
