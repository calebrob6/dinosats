# DinoSats

**Jump to: [Using the app](#using-the-app) | [How it works](#how-it-works) | [Run it locally](#run-it-locally) | [Make your own scene](#make-your-own-scene) | [Configuration](#configuration) | [Hosting](#hosting) | [Credits](#credits) | [Citation](#citation) | [License](#license)**

Search aerial imagery with words. Type *runway*, *marina*, *solar panels* or *aircraft carrier*, and DinoSats shows where they are across 380 km² of San Diego, in your browser. It compares precomputed [DINOv3](https://github.com/facebookresearch/dinov3) embeddings of the imagery with your prompt, encoded by [dino.txt](https://arxiv.org/abs/2412.16334), a text model trained to align with DINOv3.

**Try it: [calebrob.com/dinosats](https://calebrob.com/dinosats/)**

<p align="center">
  <a href="https://calebrob.com/dinosats/"><img src="web/teaser.jpg" alt="DinoSats teaser: a heatmap for the prompt beach over San Diego, tracing the ocean beaches of Coronado and the Silver Strand" width="100%"></a>
</p>

*The heatmap for the prompt* beach *highlights the ocean beaches of Coronado and the Silver Strand.*

## Using the app

- **Search**: type a prompt (`/` moves the cursor to the search box) or click an example. About 900 common prompts are precomputed. Anything else downloads a 664 MB text encoder the first time, which the browser normally caches.
- **Top matches**: switch this on to list the 10 highest-scoring places, and click one to zoom to it. They're chosen from a coarse (about 20 m) version of the whole scene, so small objects may only appear after you zoom in.
- **Describe a location**: `Shift` + click shows the words that best match that location.
- **Display**: switch between the relative score and raw similarity, and change the threshold, opacity and background prompts. `Space` hides or shows the heatmap.

## How it works

DINOv3, an image model from Meta, turns each patch of imagery (about 5 m across) into a vector, a list of numbers that describes what the patch looks like. Its companion text model, dino.txt, was trained to turn text into vectors of the same kind, so the vector for a patch of runway is similar to the vector for the word *runway*. The app scores each patch by how similar its vector is to your prompt's.

- **Ahead of time**, we embed [NAIP](https://planetarycomputer.microsoft.com/dataset/naip) aerial imagery (0.6 m per pixel, 2022) with DINOv3 and compress each patch to 64 bytes. San Diego has 16.5 million patches, stored in a 1.1 GB Cloud-Optimized GeoTIFF.
- **In your browser**, the text model runs on your own device with onnxruntime-web. There is no backend server; the site consists of static files. The map loads a coarse version of the embeddings first and downloads higher-resolution blocks as you zoom in.
- **Scores are relative.** Each patch's score is how much better it matches your prompt than generic background prompts such as *water*, *road* and *buildings*. This makes the heatmap much easier to read than raw similarity.

It's zero-shot, and dino.txt was trained on web photos, not aerial imagery. It does best on things that are large and distinctive from above (runways, golf courses, marinas, sports fields) and worse on small or ambiguous objects. It is a quick way to explore a scene, not a replacement for a trained detector.

## Run it locally

```bash
git clone https://github.com/calebrob6/dinosats.git
cd dinosats
npm install
npm run build
python3 serve.py   # open http://localhost:8766/
```

You'll need Node.js and Python 3. This uses the same hosted data as the demo. Browsers only allow the model cache and multi-threading on `localhost` or https, so on a remote machine, forward the port with `ssh -L 8766:localhost:8766 <host>`.

## Make your own scene

Start from the clone in [Run it locally](#run-it-locally). You'll also need a CUDA GPU, about 34 GB of RAM (for San Diego), Python 3.12, [azcopy](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10), and access to the gated [DINOv3 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) on Hugging Face.

**Set up** the environment and download the models:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git

.venv/bin/hf auth login
.venv/bin/hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir weights/hf-dinov3-vitl16-lvd1689m
.venv/bin/hf download PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder \
    dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth --local-dir weights
curl -L -o weights/bpe_simple_vocab_16e6.txt.gz https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz
```

The dino.txt weights on Hugging Face are a mirror of the file Meta distributes through the [dinov3 repo](https://github.com/facebookresearch/dinov3#pretrained-heads---zero-shot-tasks-with-dinotxt).

**Build the scene.** These are the commands for San Diego. For another area, replace the NAIP quarter quad IDs and file names with your own.

```bash
# 1. Download the imagery
.venv/bin/python scripts/download_naip.py 3211715_sw 3211715_se 3211716_sw 3211723_nw 3211723_ne 3211724_nw 3211723_sw 3211723_se 3211724_sw --year 2022 --state ca

# 2. Mosaic it
mkdir -p web/data
.venv/bin/python scripts/make_mosaic.py data/naip/*.tif --out data/naip_2022_san_diego_3x3.tif --web_out web/data/naip_2022_san_diego_3x3_rgb.tif

# 3. Embed it (the slow step: about an hour on two GPUs; use --device cuda:0 for one)
.venv/bin/python scripts/embed.py --input_fn data/naip_2022_san_diego_3x3.tif --output_fn web/data/naip_2022_san_diego_3x3_dinotxt_pca64_2x_s256.tif --resize_factor 2 --device cuda:0,cuda:1

# 4. Precompute the common prompts
.venv/bin/python scripts/embed_vocab.py

# 5. Export the text encoder for the browser
.venv/bin/python scripts/export_text_encoder.py

# 6. Copy the config with local paths (edit the file names if yours differ), then open http://localhost:8766/?config=data/local.json
sed -E 's#https://[^"]*/dinosats/##' web/config.json > web/data/local.json
python3 serve.py
```

A quad ID such as `3211723_ne` means the north-east quarter of quad 23 in the 1° cell at 32° N, 117° W, with quads numbered 1 to 64 from the north-west corner. Every script takes `--help`.

## Configuration

The app reads [`web/config.json`](web/config.json), which lists the scene's files, the text model, the background prompts and the example prompts. Paths can be relative to `web/` or full URLs.

<details>
<summary>All fields and URL options</summary>

| Field | What it is |
|:--|:--|
| `scene.name`, `scene.subtitle` | Place name and imagery label shown in the app |
| `scene.rgb` | RGB imagery COG, from `make_mosaic.py --web_out` |
| `scene.embeddings`, `scene.pca` | Embedding COG and its `.pca.json`, from `embed.py` |
| `scene.posbasis`, `scene.posnorm` | Optional, from `embed.py`: removes DINOv3's positional bias, a faint repeating pattern on uniform areas like open water |
| `model`, `vocab` | ONNX text encoder and CLIP tokenizer vocabulary |
| `background` | Background prompts for the relative score |
| `examples` | Example prompts under the search box |
| `point_vocab` | Optional precomputed prompts, from `embed_vocab.py` |

URL options: `?config=<url>` loads another config, `?debug=true` shows developer controls, `?debias=0` keeps the positional bias, and `?ep=wasm` skips WebGPU.

</details>

## Hosting

The app consists only of static files, so you can host the code, data and model on any cloud object store or web server. The host needs to serve https and support HTTP Range requests, since the app reads the GeoTIFFs block by block, and if it's on a different origin from the app, send CORS headers that expose `Content-Range`. The demo's code is deployed to GitHub Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`.

To check a running copy end to end, run `npx playwright install chromium` once, then `node scripts/smoke_test.mjs [url]`.

## Credits

- Imagery: [NAIP](https://planetarycomputer.microsoft.com/dataset/naip) (USDA), via the Microsoft Planetary Computer. Public domain.
- Models: [DINOv3 and dino.txt](https://github.com/facebookresearch/dinov3) from Meta, under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- Tokenizer: the CLIP BPE vocabulary, redistributed by Meta under its [MIT license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE).
- Positional bias removal: the method from [INSID3](https://github.com/visinf/INSID3).

## Citation

If you use DinoSats, please cite it along with [DINOv3](https://github.com/facebookresearch/dinov3#citing-dinov3) and [dino.txt](https://arxiv.org/abs/2412.16334):

```bibtex
@misc{robinson2026dinosats,
  author       = {Robinson, Caleb},
  title        = {{DinoSats}: open-vocabulary search over aerial imagery in the browser},
  year         = {2026},
  howpublished = {\url{https://github.com/calebrob6/dinosats}}
}
```

## License

The code is MIT licensed (see [`LICENSE`](LICENSE)). The DINOv3 weights and everything derived from them, including the hosted embeddings and text encoder, are under the DINOv3 License.
