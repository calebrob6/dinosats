// Port of the CLIP SimpleTokenizer used by dino.txt (dinov3/thirdparty/CLIP/clip/simple_tokenizer.py).

const PATTERN = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
const LIGATURES = { "\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl", "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "ſt", "\uFB06": "st" };

// Returns [byte -> unicode char lookup, chars in vocab order] (vocab order lists printable bytes first, like CLIP).
function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  const lookup = new Array(256);
  bs.forEach((b, i) => (lookup[b] = String.fromCodePoint(cs[i])));
  return [lookup, cs.map((c) => String.fromCodePoint(c))];
}

// The parts of ftfy.fix_text that matter for typed prompts.
function fixText(text) {
  return text
    .replace(/\r\n|[\r\u0085\u2028\u2029]/g, "\n")
    .replace(/[\u0000-\u0008\u000B\u000E-\u001F\u007F\uFEFF]/g, "")
    .replace(/[\u2018\u2019\u201A\u201B\u2032]/g, "'")
    .replace(/[\u201C\u201D\u201E\u201F\u2033]/g, '"')
    .replace(/[\uFB00-\uFB06]/g, (c) => LIGATURES[c])
    .replace(/[\uFF01-\uFF5E]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xfee0))
    .replace(/\u3000/g, " ")
    .normalize("NFC");
}

const NAMED_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: "\u00A0" };

function htmlUnescape(text) {
  if (!text.includes("&")) return text;
  if (typeof document !== "undefined") {
    const el = document.createElement("textarea");
    el.innerHTML = text;
    return el.value;
  }
  return text.replace(/&(#x[0-9a-f]+|#[0-9]+|[a-z]+);/gi, (m, e) => {
    if (e[0] === "#") return String.fromCodePoint(parseInt(e[1].toLowerCase() === "x" ? e.slice(2) : e.slice(1), e[1].toLowerCase() === "x" ? 16 : 10));
    return NAMED_ENTITIES[e.toLowerCase()] ?? m;
  });
}

export class ClipTokenizer {
  constructor(vocabText, contextLength = 77) {
    this.contextLength = contextLength;
    const [byteEncoder, byteChars] = bytesToUnicode();
    this.byteEncoder = byteEncoder;
    const merges = vocabText.split("\n").slice(1, 49152 - 256 - 2 + 1).map((m) => m.split(" "));
    const vocab = [...byteChars, ...byteChars.map((v) => v + "</w>"), ...merges.map((m) => m.join(""))];
    vocab.push("<|startoftext|>", "<|endoftext|>");
    this.encoder = new Map(vocab.map((v, i) => [v, i]));
    this.bpeRanks = new Map(merges.map((m, i) => [m[0] + " " + m[1], i]));
    this.sot = this.encoder.get("<|startoftext|>");
    this.eot = this.encoder.get("<|endoftext|>");
    this.cache = new Map([["<|startoftext|>", ["<|startoftext|>"]], ["<|endoftext|>", ["<|endoftext|>"]]]);
    this.utf8 = new TextEncoder();
  }

  static async load(url, contextLength = 77) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
    return new ClipTokenizer(await res.text(), contextLength);
  }

  bpe(token) {
    const cached = this.cache.get(token);
    if (cached) return cached;
    let word = [...token];
    word[word.length - 1] += "</w>";
    while (word.length > 1) {
      let best = null;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.bpeRanks.get(word[i] + " " + word[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = [word[i], word[i + 1]];
        }
      }
      if (!best) break;
      const merged = [];
      for (let i = 0; i < word.length; ) {
        if (i < word.length - 1 && word[i] === best[0] && word[i + 1] === best[1]) {
          merged.push(best[0] + best[1]);
          i += 2;
        } else {
          merged.push(word[i++]);
        }
      }
      word = merged;
    }
    this.cache.set(token, word);
    return word;
  }

  encode(text) {
    text = htmlUnescape(htmlUnescape(fixText(text))).trim();
    text = text.replace(/\s+/gu, " ").trim().toLowerCase();
    const ids = [];
    for (const [token] of text.matchAll(PATTERN)) {
      const mapped = Array.from(this.utf8.encode(token), (b) => this.byteEncoder[b]).join("");
      for (const piece of this.bpe(mapped)) ids.push(this.encoder.get(piece));
    }
    return ids;
  }

  // [SOT, ...tokens, EOT], truncated like the Python tokenizer but not padded.
  tokenize(text) {
    const ids = [this.sot, ...this.encode(text), this.eot];
    if (ids.length > this.contextLength) {
      ids.length = this.contextLength;
      ids[ids.length - 1] = this.eot;
    }
    return ids;
  }
}
