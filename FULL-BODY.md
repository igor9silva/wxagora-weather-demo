# Full-body weather presenter: experimental local pipeline

This path uses an audio-conditioned video model to generate the whole presenter in motion, rather than altering a mouth on a static photograph. The included presenter is a fictional image generated with Codex's built-in image tool. The four background plates are generated images, not photographs or live views, and their weather is based on the saved IPMA daily forecast.

The prototype does **not** promise accurate lip sync, natural gestures, or correct pronunciation. Render a short scene and review it before investing time in a full run or publishing. No automatic social-media posting is included. The generated images use Codex's image tool; the voice and video inference run locally.

## Prepare a day's forecast and narration

On macOS with Python 3.11 and FFmpeg, install the local speech dependencies in a virtual environment:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --pre 'tts_eu_pt==0.1.1' Pillow numpy soundfile num2words
.venv/bin/python prepare_daily.py --today --out work/today --speech
```

The `plan.json` includes four image prompts and spoken text, derived from the city IDs, forecast date, weather codes and maximum temperatures in the IPMA response. The program rejects missing or unknown weather information and a mismatched current date. It does not call a paid image API. Give each prompt to Codex's built-in image-generation tool and save its result to the indicated `image_file` name in a date-specific image directory; then register those images:

```sh
.venv/bin/python register_image_set.py --plan work/today/plan.json --images assets/generated/YYYY-MM-DD
```

The registration records each forecast, exact prompt and image hash. The renderer refuses images when the date, weather code, temperature or image hash no longer matches the plan. Codex can orchestrate the image-generation step as part of a future task; this repository does not itself provide unattended access to the built-in image tool, and no daily schedule is installed.

## Local audio-to-video model

This experiment uses [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) and the 4-bit LTX-2.3 conversion on Apple Silicon. The model weights have their own [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE), distinct from the port's MIT code licence. Approximately 29 GB must be downloaded once for the selected video weights and text encoder; the separate voice model also needs its first download.

For the tested port revision, set up a separate environment and download only the files needed for this path:

```sh
git clone https://github.com/dgrauet/ltx-2-mlx.git work/ltx-2-mlx
git -C work/ltx-2-mlx checkout -b wxagora-compat 895201df379b079c2154738bcf92005b7a371923
cd work/ltx-2-mlx
uv sync --no-dev
.venv/bin/hf download dgrauet/ltx-2.3-mlx-q4 \
  config.json embedded_config.json quantize_config.json split_model.json \
  transformer-distilled-1.1.safetensors connector.safetensors audio_vae.safetensors \
  vae_encoder.safetensors vae_decoder.safetensors vocoder.safetensors \
  spatial_upscaler_x2_v1_1.safetensors spatial_upscaler_x2_v1_1_config.json \
  --local-dir ../models/ltx-2.3-q4
.venv/bin/hf download mlx-community/gemma-3-12b-it-4bit --local-dir ../models/gemma-3-12b-q4
cd ../..
```

Use the plan, matching generated images and an authorized full-body green-screen portrait. This renders one city as a test; remove `--only porto` to process the other scenes as well:

```sh
.venv/bin/python render_fullbody.py \
  --plan work/today/plan.json --images assets/generated/YYYY-MM-DD \
  --presenter assets/generated/presenter-green-screen.png \
  --ltx work/ltx-2-mlx/.venv/bin/ltx-2-mlx \
  --model work/models/ltx-2.3-q4 --gemma work/models/gemma-3-12b-q4 \
  --only porto --output work/porto-test.mp4
```

The audio-to-video mode here is marked experimental by its upstream port, and its 2.3-weight variant was not listed as validated there. The code pads the source audio instead of trimming it to fit the model's frame grid, and refuses to reuse clips whose audio, source portrait or output bytes changed. When assembling the cities, it joins the original voice tracks at exact frame boundaries and encodes the final audio once; copying already encoded AAC segments can accumulate audible delays at the cuts. `--reuse-clips` is intended only for recomposing matching previously generated clips. Model weights, inference time, hardware and electricity are not included in any claim of a zero total cost.

## Reference presenter prompt

The included fictional reference was generated with Codex's built-in image tool from this prompt:

> Use case: photorealistic-natural. A production reference frame for a fictional Portuguese TV weather reporter, adult woman in her early thirties, medium-long dark brown hair, wearing a tailored charcoal blazer and matching trousers with a white top and dark shoes. She is standing in a relaxed, approachable full-body pose, facing the camera, holding a small wireless microphone in one hand and making a subtle natural presenting gesture with the other. Full body from top of hair to shoes must be completely visible with generous margin around the feet and head, and both hands must be visible. Static camera, portrait-format 9:16 composition. Even studio lighting. The entire background and floor are a uniform flat bright chroma green screen for a later video composite; no green clothes. No text, numbers, logos, watermarks, scenery, objects or additional people. This is a fictional person, not a copy of an existing person.
