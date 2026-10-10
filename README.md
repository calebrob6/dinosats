# DinoSats

**Jump to: [Using the app](#using-the-app) | [How it works](#how-it-works) | [Run it locally](#run-it-locally) | [Make your own scene](#make-your-own-scene) | [Configuration](#configuration) | [Hosting](#hosting) | [Credits](#credits) | [Citation](#citation) | [License](#license)**

The DinoSats app lets you search aerial imagery by typing a description of what you are looking for. For example, if you type *runway*, *marina*, *solar panels* or *aircraft carrier*, the app shows where those things are across 380 km² of San Diego. Everything runs in your web browser. The app compares precomputed [DINOv3](https://github.com/facebookresearch/dinov3) embeddings of the imagery with an embedding of your prompt, which is computed by [dino.txt](https://arxiv.org/abs/2412.16334), a text model that was trained to align with DINOv3.

**You can try the app at [calebrob.com/dinosats](https://calebrob.com/dinosats/).**

<p align="center">
  <a href="https://calebrob.com/dinosats/"><img src="web/teaser.jpg" alt="DinoSats teaser: a heatmap for the prompt beach over San Diego, highlighting the ocean beaches of Coronado and the Silver Strand" width="100%"></a>
</p>

The image above shows the heatmap for the prompt *beach*, which highlights the ocean beaches of Coronado and the Silver Strand.

## Using the app

- **Search.** Type a prompt in the search box, or click one of the example prompts below it. Pressing `/` moves the cursor to the search box. About 900 common prompts are precomputed. Any other prompt needs the 664 MB text encoder, which the app downloads the first time you use it and the browser normally caches for later visits.
- **Top matches.** When you switch on Top matches, the app lists the 10 highest-scoring places, and clicking one zooms the map to it. The list is chosen from a coarse version of the whole scene, with patches about 20 m across, so small objects may only appear after you zoom in.
- **Describe a location.** If you hold `Shift` and click on the map, the app shows the words whose embeddings best match that location.
- **Display settings.** The Display card lets you switch between the relative score and raw similarity, and change the threshold, the opacity and the background prompts. Pressing `Space` hides or shows the heatmap.

## How it works

DINOv3 is an image model from Meta. It turns each patch of imagery, about 5 m across, into a vector: a list of numbers that describes what the patch looks like. Its companion text model, dino.txt, was trained to turn text into vectors of the same kind, so the vector for a patch of runway is similar to the vector for the word *runway*. The app scores each patch by how similar its vector is to the vector for your prompt.

- **Embedding the imagery ahead of time.** We embed 2022 [NAIP](https://planetarycomputer.microsoft.com/dataset/naip) aerial imagery, which has 0.6 m pixels, with DINOv3, and we compress each patch's vector to 64 bytes. The San Diego scene has 16.5 million patches, which are stored in a 1.1 GB Cloud-Optimized GeoTIFF.
- **Encoding the prompt in your browser.** The text model runs on your own device with onnxruntime-web, so there is no backend server, and the site consists only of static files.
- **Loading the embeddings.** The map loads a coarse version of the embeddings first, so the app is usable within a few seconds, and then downloads higher-resolution blocks as you zoom in.
- **Scoring against background prompts.** Each patch's score is how much better it matches your prompt than a set of generic background prompts, such as *water*, *road* and *buildings*. These relative scores make the heatmap much easier to read than the raw similarity.

DinoSats is zero-shot: dino.txt was trained on web photos, not aerial imagery, and we have not fine-tuned it. It works best on things that are large and distinctive when seen from above, such as runways, golf courses, marinas and sports fields, and it works less well on small or ambiguous objects. It is a quick way to explore a scene, not a replacement for a trained detector.

## Run it locally

To run the app on your own machine with the same hosted data as the demo, you need Node.js and Python 3. Clone the repo, build the app and start the local server:

```bash
git clone https://github.com/calebrob6/dinosats.git
cd dinosats
npm install
npm run build
python3 serve.py
```

Then open http://localhost:8766/ in your browser. Browsers only allow the model cache and multi-threading on `localhost` or over https, so if you run the server on a remote machine, forward the port with `ssh -L 8766:localhost:8766 <host>` and open the app through the forwarded port.

## Make your own scene

To build a scene for a different area, start from the clone in [Run it locally](#run-it-locally). You also need a CUDA GPU, about 34 GB of RAM for a scene the size of San Diego, Python 3.12, [azcopy](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10), and access to the gated [DINOv3 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) on Hugging Face.

First, set up a Python environment and download the models:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git

.venv/bin/hf auth login
.venv/bin/hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir weights/hf-dinov3-vitl16-lvd1689m
.venv/bin/hf download PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder \
    dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth --local-dir weights
curl -L -o weights/bpe_simple_vocab_16e6.txt.gz https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz
```

The dino.txt weights on Hugging Face are a mirror of the file that Meta distributes through the [dinov3 repo](https://github.com/facebookresearch/dinov3#pretrained-heads---zero-shot-tasks-with-dinotxt).

Next, build the scene. The commands below build the San Diego scene. For a different area, replace the NAIP quarter quad IDs and the file names with your own.

```bash
# 1. Download the NAIP quarter quads.
.venv/bin/python scripts/download_naip.py 3211715_sw 3211715_se 3211716_sw 3211723_nw 3211723_ne 3211724_nw 3211723_sw 3211723_se 3211724_sw --year 2022 --state ca

# 2. Mosaic them into one GeoTIFF for embedding and one RGB GeoTIFF for display.
mkdir -p web/data
.venv/bin/python scripts/make_mosaic.py data/naip/*.tif --out data/naip_2022_san_diego_3x3.tif --web_out web/data/naip_2022_san_diego_3x3_rgb.tif

# 3. Embed the mosaic with DINOv3. This is the slowest step: it takes about an hour on two GPUs. To use one GPU, pass --device cuda:0.
.venv/bin/python scripts/embed.py --input_fn data/naip_2022_san_diego_3x3.tif --output_fn web/data/naip_2022_san_diego_3x3_dinotxt_pca64_2x_s256.tif --resize_factor 2 --device cuda:0,cuda:1

# 4. Precompute the embeddings of the common prompts.
.venv/bin/python scripts/embed_vocab.py

# 5. Export the text encoder to ONNX for the browser.
.venv/bin/python scripts/export_text_encoder.py

# 6. Copy the config and replace the hosted URLs with local paths. If your file names differ, edit them in local.json.
sed -E 's#https://[^"]*/dinosats/##' web/config.json > web/data/local.json
python3 serve.py
```

Finally, open http://localhost:8766/?config=data/local.json to see your scene.

A quad ID such as `3211723_ne` means the north-east quarter of quad 23 in the 1° cell at 32° N, 117° W. Quads are numbered 1 to 64, row by row, starting from the north-west corner of the cell. Every script accepts `--help`, which lists its options.

## Configuration

The app reads its settings from [`web/config.json`](web/config.json), which lists the scene's files, the text model, the background prompts and the example prompts. Each path can be relative to `web/` or a full URL.

<details>
<summary>All configuration fields and URL options</summary>

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

The app also accepts these URL options: `?config=<url>` loads a different config file, `?debug=true` shows developer controls, `?debias=0` keeps the positional bias, and `?ep=wasm` runs the text model on WebAssembly instead of WebGPU.

</details>

## Hosting

The app consists only of static files, so you can host the code, the data and the model on any cloud object store or web server. The server needs to use https and support HTTP Range requests, because the app reads the GeoTIFFs one block at a time. If the data is on a different origin from the app, the server also needs to send CORS headers that expose `Content-Range`. The code for the demo is deployed to GitHub Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`.

## Credits

- The imagery is from [NAIP](https://planetarycomputer.microsoft.com/dataset/naip) (USDA), accessed through the Microsoft Planetary Computer, and is in the public domain.
- The [DINOv3 and dino.txt](https://github.com/facebookresearch/dinov3) models are from Meta and are released under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- The CLIP BPE tokenizer vocabulary is redistributed by Meta under its [MIT license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE).
- The method for removing the positional bias is from [INSID3](https://github.com/visinf/INSID3).

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
