// Scoring dino.txt text embeddings against the uint8 PCA embedding COG written by scripts/embed.py.
// Pure functions on typed arrays, so this runs unchanged in the browser and in Node tests.

export class PcaProjection {
  /** @param {object} pca contents of the <output>.pca.json written by scripts/embed.py */
  constructor(pca) {
    this.k = pca.n_components;
    this.dim = pca.input_dim;
    this.mean = Float32Array.from(pca.mean);
    this.components = Float32Array.from(pca.components.flat());
    const { vmin, vmax } = pca.quantization;
    this.vmin = Float32Array.from(vmin);
    this.step = Float32Array.from(vmax, (v, j) => (v - vmin[j]) / 255);
  }

  /**
   * Folds the PCA projection and the dequantisation into per-code weights, so that for a token with uint8 codes q,
   * cos(t, x) ≈ bias + Σ_j w[j] q[j]. t: L2-normalised text embedding (dim).
   */
  weights(t) {
    const { k, dim, components } = this;
    const w = new Float32Array(k);
    let bias = 0;
    for (let d = 0; d < dim; d++) bias += this.mean[d] * t[d];
    for (let j = 0; j < k; j++) {
      let u = 0;
      for (let d = 0, off = j * dim; d < dim; d++) u += components[off + d] * t[d];
      bias += u * this.vmin[j];
      w[j] = u * this.step[j];
    }
    return { bias, w };
  }

  /** Dequantised, PCA-reconstructed (unnormalised) patch embedding of token i of pixel-interleaved codes. */
  pixelEmbedding(codes, i) {
    const { k, dim, components } = this;
    const x = Float32Array.from(this.mean);
    for (let j = 0; j < k; j++) {
      const z = this.vmin[j] + codes[i * k + j] * this.step[j];
      for (let d = 0, off = j * dim; d < dim; d++) x[d] += z * components[off + d];
    }
    return x;
  }
}

/**
 * Positional debiasing (INSID3): project the positional subspace B out of both the patch embedding and the query,
 * P = I - B^T B, and compare the renormalised results. P is symmetric and idempotent, so
 * cos(P t, P x) = (P t / |P t|) . x / |P x|: only the query is projected, and the per-token norm comes from the PCA
 * codes (z is the dequantised code): |P x|^2 = a + 2 b.z + z^T G z with G = C P C^T, b = C P mean and a = mean^T P mean.
 */
export class PositionalDebias {
  /** @param {Float32Array} basis nb * dim floats, orthonormal rows  @param {PcaProjection} pca */
  constructor(basis, pca) {
    const { k, dim, mean, components: C } = pca;
    if (basis.length % dim) throw new Error(`positional basis length ${basis.length} is not a multiple of ${dim}`);
    this.pca = pca;
    this.basis = basis;
    this.nb = basis.length / dim;
    const { nb } = this;
    const dot = (a, ao, b, bo) => {
      let u = 0;
      for (let d = 0; d < dim; d++) u += a[ao + d] * b[bo + d];
      return u;
    };
    const BC = new Float64Array(nb * k);
    const Bm = new Float64Array(nb);
    for (let r = 0; r < nb; r++) {
      Bm[r] = dot(basis, r * dim, mean, 0);
      for (let j = 0; j < k; j++) BC[r * k + j] = dot(basis, r * dim, C, j * dim);
    }
    // Augmented (k + 1) x (k + 1) quadratic form over (z, 1), upper triangle only (off-diagonal pairs counted twice).
    const m = k + 1;
    const Q = new Float64Array(m * m);
    for (let j = 0; j < k; j++) {
      for (let l = j; l < k; l++) {
        let g = dot(C, j * dim, C, l * dim);
        for (let r = 0; r < nb; r++) g -= BC[r * k + j] * BC[r * k + l];
        Q[j * m + l] = g;
      }
      let bj = dot(C, j * dim, mean, 0);
      for (let r = 0; r < nb; r++) bj -= BC[r * k + j] * Bm[r];
      Q[j * m + k] = bj;
    }
    let a = dot(mean, 0, mean, 0);
    for (let r = 0; r < nb; r++) a -= Bm[r] * Bm[r];
    Q[k * m + k] = a;
    for (let j = 0; j < m; j++) for (let l = j + 1; l < m; l++) Q[j * m + l] *= 2;
    this.Q = Float32Array.from(Q);
    this.norms = new WeakMap();
  }

  /** P t / |P t|: the debiased query, to score against tokens scaled by invNorm. */
  query(t) {
    const out = this.projectOut(t);
    let sq = 0;
    for (const v of out) sq += v * v;
    const n = Math.sqrt(sq) || 1;
    for (let d = 0; d < out.length; d++) out[d] /= n;
    return out;
  }

  /** |P t| = sqrt(|t|^2 - |B t|^2), cached per embedding (for shift-click lookups over the whole vocabulary). */
  textNorm(t) {
    let n = this.norms.get(t);
    if (n === undefined) {
      const { basis, nb } = this;
      const dim = this.pca.dim;
      let sq = 0;
      for (let d = 0; d < dim; d++) sq += t[d] * t[d];
      for (let r = 0, off = 0; r < nb; r++, off += dim) {
        let u = 0;
        for (let d = 0; d < dim; d++) u += basis[off + d] * t[d];
        sq -= u * u;
      }
      n = Math.sqrt(Math.max(sq, 1e-6));
      this.norms.set(t, n);
    }
    return n;
  }

  /** Fills the textNorm cache for many embeddings without blocking the page. */
  async warmTextNorms(embeddings) {
    let lastYield = performance.now();
    for (const t of embeddings) {
      this.textNorm(t);
      if (performance.now() - lastYield > 30) {
        await new Promise((resolve) => setTimeout(resolve));
        lastYield = performance.now();
      }
    }
  }

  /** t - B^T (B t) */
  projectOut(t) {
    const { basis, nb } = this;
    const dim = this.pca.dim;
    const out = Float32Array.from(t);
    for (let r = 0, off = 0; r < nb; r++, off += dim) {
      let u = 0;
      for (let d = 0; d < dim; d++) u += basis[off + d] * t[d];
      for (let d = 0; d < dim; d++) out[d] -= u * basis[off + d];
    }
    return out;
  }

  /** 1 / |P x| for every token of pixel-interleaved codes; yields to the event loop if `yieldMs` is set. */
  async invNorm(codes, onProgress) {
    const { k, vmin, step } = this.pca;
    const m = k + 1;
    const { Q } = this;
    const n = codes.length / k;
    const out = new Float32Array(n);
    const zt = new Float32Array(m);
    zt[k] = 1;
    let lastYield = performance.now();
    for (let i = 0; i < n; i++) {
      for (let j = 0, b = i * k; j < k; j++) zt[j] = vmin[j] + codes[b + j] * step[j];
      let sum = 0;
      for (let j = 0; j < m; j++) {
        const row = j * m;
        let u = 0;
        for (let l = j; l < m; l++) u += Q[row + l] * zt[l];
        sum += zt[j] * u;
      }
      out[i] = 1 / Math.sqrt(Math.max(sum, 1e-6));
      if (i % 20000 === 19999 && performance.now() - lastYield > 40) {
        onProgress?.(i / n);
        await new Promise((resolve) => setTimeout(resolve));
        lastYield = performance.now();
      }
    }
    return out;
  }
}

/** cos(t, x) for every token of pixel-interleaved uint8 codes, given weights from PcaProjection.weights(t). */
export function cosine(codes, { bias, w }) {
  const k = w.length;
  const n = codes.length / k;
  const out = new Float32Array(n);
  for (let i = 0, b = 0; i < n; i++, b += k) {
    let s = bias;
    for (let j = 0; j < k; j++) s += w[j] * codes[b + j];
    out[i] = s;
  }
  return out;
}

/** Per-pixel log-sum-exp of scale * cos over a list of score arrays. */
export function logSumExp(scores, scale) {
  const n = scores[0].length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (const s of scores) m = Math.max(m, scale * s[i]);
    let z = 0;
    for (const s of scores) z += Math.exp(scale * s[i] - m);
    out[i] = m + Math.log(z);
  }
  return out;
}

/** Softmax probability of the query against the background prompts: sigmoid(scale * q - lse(background)). */
export function relativeProbability(query, backgroundLse, scale) {
  const out = new Float32Array(query.length);
  for (let i = 0; i < query.length; i++) out[i] = 1 / (1 + Math.exp(backgroundLse[i] - scale * query[i]));
  return out;
}

/** Approximate quantiles from a strided sample. */
export function quantiles(values, qs, sampleSize = 50000) {
  const stride = Math.max(1, Math.floor(values.length / sampleSize));
  const sample = new Float32Array(Math.ceil(values.length / stride));
  for (let i = 0, j = 0; i < values.length; i += stride, j++) sample[j] = values[i];
  sample.sort();
  return qs.map((q) => sample[Math.min(sample.length - 1, Math.max(0, Math.round(q * (sample.length - 1))))]);
}

/**
 * Greedy peak picking: highest values first, skipping anything within `radius` cells (Chebyshev) of a kept peak.
 * Returns [{index, row, col, value}].
 */
export function findPeaks(values, width, { count = 10, radius = 8, min = 0.5, maxCandidates = 200000 } = {}) {
  let idx = [];
  for (let i = 0; i < values.length; i++) if (values[i] >= min) idx.push(i);
  idx.sort((a, b) => values[b] - values[a]);
  if (idx.length > maxCandidates) idx = idx.slice(0, maxCandidates);
  const peaks = [];
  for (const i of idx) {
    const row = Math.floor(i / width);
    const col = i - row * width;
    if (peaks.some((p) => Math.abs(p.row - row) <= radius && Math.abs(p.col - col) <= radius)) continue;
    peaks.push({ index: i, row, col, value: values[i] });
    if (peaks.length >= count) break;
  }
  return peaks;
}

/** Average the L2-normalised [start, end) slice of each row of a (batch, dim) embedding, then renormalise. */
export function ensembleEmbedding(data, batch, [start, end]) {
  const dim = data.length / batch;
  const out = new Float32Array(end - start);
  for (let b = 0; b < batch; b++) {
    const row = data.subarray(b * dim + start, b * dim + end);
    let norm = 0;
    for (const v of row) norm += v * v;
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < row.length; d++) out[d] += row[d] / norm;
  }
  let norm = 0;
  for (const v of out) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  for (let d = 0; d < out.length; d++) out[d] /= norm;
  return out;
}

/**
 * Greedy retrieval of the strings whose text embeddings best match one pixel's patch embedding.
 * @param {Float32Array} x patch embedding (e.g. EmbeddingIndex.pixelEmbedding)
 * @param {{strings: string[], embeddings: Float32Array[]}[]} sources candidate sets of L2-normalised text embeddings
 * @param {((t: Float32Array) => number)?} textNorm divides each text's cosine (|P t| when x is debiased, P x)
 * @returns {{text: string, cos: number, prob: number}[]} top `count` by cosine; prob is a softmax over all candidates
 */
export function topStrings(x, sources, scale, count = 10, textNorm = null) {
  let norm = 0;
  for (const v of x) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  const seen = new Set();
  const all = [];
  for (const { strings, embeddings } of sources) {
    for (let i = 0; i < strings.length; i++) {
      if (seen.has(strings[i])) continue;
      seen.add(strings[i]);
      const t = embeddings[i];
      let dot = 0;
      for (let d = 0; d < x.length; d++) dot += t[d] * x[d];
      all.push({ text: strings[i], cos: dot / norm / (textNorm ? textNorm(t) : 1) });
    }
  }
  if (!all.length) return [];
  const m = Math.max(...all.map((a) => a.cos));
  let z = 0;
  for (const a of all) z += Math.exp(scale * (a.cos - m));
  for (const a of all) a.prob = Math.exp(scale * (a.cos - m)) / z;
  return all.sort((a, b) => b.cos - a.cos).slice(0, count);
}
