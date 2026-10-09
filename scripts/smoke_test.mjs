// End-to-end check of the search app in headless Chromium: loads the page, runs queries, checks the heatmap and
// top matches, and saves a screenshot. Start `python3 serve.py` first, then: node scripts/smoke_test.mjs [url] [out.png]
import { chromium } from "playwright";

const url = new URL(process.argv[2] ?? "http://localhost:8766/");
url.searchParams.set("debug", "true"); // shows the positional-bias toggle tested below
const screenshot = process.argv[3] ?? "smoke_test.png";
// Example queries come from the precomputed vocabulary; the last one needs the in-browser text encoder.
const queries = ["runway", "ships", "train yard"];
const novel = "cruise ship docked at a pier";

const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));

let ok = true;
const t0 = Date.now();
await page.goto(url.href);
// On hosts without COOP/COEP headers, coi-serviceworker reloads the page once after installing.
for (;;) {
  try {
    await page.waitForFunction(() => document.getElementById("status")?.textContent === "Ready", null, { timeout: 300_000 });
    break;
  } catch (err) {
    if (!/context was destroyed|navigat/i.test(err.message)) throw err;
  }
}
const ready = Date.now() - t0;
await page.waitForFunction(() => window.__app.state.globalFinal, null, { timeout: 120_000 });
console.log(`ready in ${(ready / 1000).toFixed(1)}s, full-resolution global level in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

// The help tour opens on the first visit; step through it, opt out of it on startup, and close it.
await page.waitForSelector(".tut-card.is-active", { timeout: 10_000 });
const tour = [];
for (let i = 0; i < 5; i++) {
  if (i === 2) await page.screenshot({ path: screenshot.replace(/(\.png)?$/, "_tour.png") });
  tour.push(await page.evaluate(() => ({ title: document.querySelector(".tut-title").textContent, body: document.querySelector(".tut-body").textContent })));
  await page.keyboard.press("ArrowRight");
}
const intro = tour[0].body;
const bytes = await page.evaluate(() => window.__app.state.embeddingBytes);
console.log("tour:", tour.map((t) => t.title).join(" | "), "|", intro.slice(0, 160), `| embeddings ${bytes} bytes`);
ok &&= tour.length === 5 && /km²/.test(intro) && /billion pixels/.test(intro) && /GB/.test(intro) && bytes > 1e9;
ok &&= !(await page.locator(".tut-card.is-active").count()); // "Get started" on the last step closes it
await page.click("#help");
await page.check(".tut-optout input");
ok &&= (await page.evaluate(() => localStorage.getItem("dinosats:hide-help"))) === "1";
await page.keyboard.press("Escape");
await page.waitForFunction(() => !document.querySelector(".tut-card.is-active"));

await page.check("#peaks-on");
const info = await page.evaluate(() => ({
  isolated: self.crossOriginIsolated,
  serviceWorker: !!navigator.serviceWorker?.controller,
  backend: window.__app.state.backend,
  grid: [window.__app.state.grid.width, window.__app.state.grid.height],
  background: window.__app.state.background.prompts.length,
}));
console.log(info);

for (const q of [...queries, novel]) {
  const t = Date.now();
  await page.fill("#query", q);
  await page.press("#query", "Enter");
  await page.waitForFunction((q) => document.getElementById("status").textContent === `Showing “${q}”`, q, { timeout: 60_000 });
  const r = await page.evaluate(() => {
    const { current, session } = window.__app.state;
    let above = 0;
    for (const v of current.values) above += v > 0.5;
    return { above, peaks: current.peaks.length, top: current.peaks[0]?.value, encoder: !!session, timing: document.getElementById("timing").textContent };
  });
  console.log(`${q}: ${Date.now() - t} ms, ${r.above} tokens with P > 0.5, ${r.peaks} peaks, top ${r.top?.toFixed(3)}, encoder loaded ${r.encoder} | ${r.timing}`);
  ok &&= r.peaks > 0 && r.encoder === (q === novel);
}

// A background prompt switches to raw cosine (with a note) until the next query that isn't one.
const modeAfter = async (q) => {
  await page.fill("#query", q);
  await page.press("#query", "Enter");
  await page.waitForFunction((q) => document.getElementById("status").textContent === `Showing “${q}”`, q, { timeout: 60_000 });
  return page.evaluate(() => ({ relative: window.__app.state.current.relative, raw: document.getElementById("mode-raw").checked, note: !document.getElementById("mode-note").hidden }));
};
const bg = await modeAfter("trees");
const back = await modeAfter("ships");
console.log("background query:", bg, "next query:", back);
ok &&= !bg.relative && bg.raw && bg.note && back.relative && !back.raw && !back.note;

// Space toggles the heatmap (the search box is blurred on submit).
const heat = [];
for (let i = 0; i < 2; i++) {
  await page.keyboard.press("Space");
  heat.push(await page.evaluate(() => [window.__app.heatVisible, document.getElementById("opacity-out").textContent]));
}
console.log("space:", JSON.stringify(heat));
ok &&= heat[0][0] === false && heat[0][1] === "off" && heat[1][0] === true && heat[1][1] !== "off";

await page.waitForTimeout(3000); // let NAIP tiles finish loading
await page.screenshot({ path: screenshot });
console.log(`screenshot: ${screenshot}`);

// Zoom to the top match: scenes whose global region is an overview should stream finer blocks for the view.
const levels = await page.evaluate(() => ({ global: window.__app.state.global.level, levels: window.__app.state.levels.map((l) => `${l.width}x${l.height}@${l.res[0].toFixed(1)}m`) }));
console.log("embedding levels", levels);
const tz = Date.now();
await page.click("#matches li a");
await page.waitForTimeout(1000);
if (levels.global > 0) {
  await page.waitForFunction(
    () => {
      const { state, fetches } = window.__app;
      return state.blocks.size > 0 && fetches.inflight.size === 0 && fetches.queue.length === 0;
    },
    null,
    { timeout: 60_000 },
  );
}
await page.waitForTimeout(3000);
const zoom = await page.evaluate(() => {
  const { state } = window.__app;
  const blocks = [...state.blocks.values()];
  return {
    blocks: blocks.length,
    scored: blocks.filter((b) => b.values?.key === state.current.key).length,
    levels: [...new Set(blocks.map((b) => b.level))],
    timing: document.getElementById("timing").textContent,
  };
});
console.log(`zoomed in ${Date.now() - tz} ms:`, zoom);
if (levels.global > 0) ok &&= zoom.scored > 0 && zoom.levels.includes(0);
const zoomShot = screenshot.replace(/(\.png)?$/, "_zoom.png");
await page.screenshot({ path: zoomShot });
console.log(`screenshot: ${zoomShot}`);

// Shift-click a point (a plain click must do nothing): the top 10 strings should be listed.
await page.mouse.click(900, 500);
await page.waitForTimeout(500);
if (await page.locator("#point-matches li").count()) throw new Error("plain click must not run the point lookup");
await page.keyboard.down("Shift");
await page.mouse.click(900, 500);
await page.keyboard.up("Shift");
await page.waitForFunction(() => document.querySelectorAll("#point-matches li").length === 10, null, { timeout: 10_000 });
const point = await page.evaluate(() => [...document.querySelectorAll("#point-matches li")].map((li) => li.textContent));
const [lat, lon] = (await page.textContent("#point-info")).split(",").map(Number);
console.log(`clicked point ${lat}, ${lon}:`, point.slice(0, 3).join(" | "));
ok &&= lat > 32 && lat < 34 && lon > -118 && lon < -116; // UTM -> lon/lat transform (San Diego)
const cards = () =>
  page.evaluate(() => ({
    matches: document.getElementById("matches-card").classList.contains("is-collapsed"),
    point: document.getElementById("point-card").classList.contains("is-collapsed"),
  }));
// With top matches on, a shift-click minimizes them.
const afterClick = await cards();
console.log("collapsed after shift-click:", afterClick);
ok &&= afterClick.matches && !afterClick.point;

// Positional debiasing: scores are recomputed for the global region and the streamed blocks.
await page.check("#debias");
await page.waitForFunction(() => /^Showing/.test(document.getElementById("status").textContent) && window.__app.state.debias, null, { timeout: 120_000 });
await page.waitForTimeout(2000);
const debiased = await page.evaluate(() => {
  const { state } = window.__app;
  const blocks = [...state.blocks.values()];
  return { global: !!state.global.inv, blocks: blocks.length, blocksOk: blocks.every((b) => b.inv), peaks: state.current.peaks.length, point: document.querySelectorAll("#point-matches li").length };
});
console.log("debiased:", debiased);
ok &&= debiased.global && debiased.blocksOk && debiased.peaks > 0 && debiased.point === 10;
await page.uncheck("#debias");
await page.waitForFunction(() => /^Showing/.test(document.getElementById("status").textContent) && !window.__app.state.debias, null, { timeout: 60_000 });

// Re-scoring (debias) leaves the cards alone; a new query minimizes the point's phrases and shows the top matches.
const afterDebias = await cards();
await page.fill("#query", "marina");
await page.press("#query", "Enter");
await page.waitForFunction(() => document.getElementById("status").textContent === "Showing “marina”", null, { timeout: 60_000 });
const afterQuery = await cards();
await page.keyboard.down("Shift");
await page.mouse.click(900, 500);
await page.keyboard.up("Shift");
await page.waitForTimeout(500);
const afterClick2 = await cards();
await page.click("#point-close");
const afterClose = await cards();
console.log("collapsed after debias:", afterDebias, "after query:", afterQuery, "after shift-click:", afterClick2, "after closing the point:", afterClose);
ok &&= afterDebias.matches && !afterDebias.point && !afterQuery.matches && afterQuery.point && afterClick2.matches && !afterClick2.point && !afterClose.matches;

if (errors.length) console.log("console errors:\n  " + errors.join("\n  "));
await browser.close();
process.exit(ok && !errors.length ? 0 : 1);
