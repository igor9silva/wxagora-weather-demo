# Portugal em viagem — WXAgora concept

Independent, non-official vertical weather videos. The first two examples use a moving 2D mascot and a stylized fictional miniature. The newer example uses a synthetic human presenter with locally generated lip motion and a different archival photograph for each forecast city. The photographs are not live views and do not show the current weather.

![Preview of the human-presenter weather video](human-preview.png)

| Video | Forecast date | Duration |
| --- | --- | --- |
| [28 October video](01-28-outubro.mp4) | 28 October 2026 | 15.3 s |
| [29 October video](02-29-outubro.mp4) | 29 October 2026 | 15.5 s |
| [Human presenter, 28 October](03-human-presenter-2026-10-28.mp4) | 28 October 2026 | 15.3 s |

The dates and weather data are saved snapshots, not a live forecast. City names, daily maximum temperatures and weather descriptions are read from `forecasts/ipma-day0.json` and `forecasts/ipma-day1.json`. Both snapshots report IPMA's source update timestamp `2026-10-28T11:31:03`. The JSON is formatted for readability, and the city and weather-type reference tables are included as well.

The human-presenter video uses its separately saved [IPMA snapshot](forecasts/ipma-human-2026-10-28.json), updated at `2026-10-28T13:31:03`.

Source: [IPMA API](https://api.ipma.pt/). The API's terms require attribution and limit free reuse to non-profit purposes. Check the current terms and appropriate licence before commercial use. This demonstration is not affiliated with WXAgora or IPMA.

## How it was made

- Portrait video: 720 × 1280, 24 fps, H.264/AAC, rendered locally with Python, Pillow and FFmpeg.
- Voice: `tts_eu_pt` 0.1.1, a Kokoro-82M model fine-tuned for European Portuguese, running locally on the Mac CPU. The original examples' clips and their exact text are in `audio/`; the daily presenter command creates fresh clips for its forecast. No macOS system voice was used in the videos.
- The short background music and transition sounds are synthesized by `render.py` from mathematical waveforms. No music catalogue or paid audio service was used.
- The older videos' illustrated landscape and mascot are one-time bitmap assets generated using Codex's built-in image-generation tool. Their prompts are in `ART-PROMPTS.md`. The human-presenter video uses archival photographs instead; their sources, creators and licences are listed in `THIRD-PARTY.md` and in the video. The media are composited locally; no video-generation API was used.

There was no metered voice or video-generation API charge for these renders. This does not establish a total cost for development, Codex/image-generation usage, electricity, hardware, data licensing or distribution. The voice model needs an approximately 313 MB one-time download if it is not already cached.

On this Mac the final video-compositing passes took approximately 6.5 and 6.0 seconds respectively. Generating the eight voice clips for both videos took about 22 seconds in one process, including model startup. These measurements exclude asset creation, initial downloads and data fetching.

A local speech recognizer recovered the temperatures but inconsistently transcribed the name “Ponta Delgada.” The voice and place-name pronunciation need a human listening review before publication.

## Render the saved examples again

On a Mac with Python 3.11 and FFmpeg installed:

```sh
cd wxagora-weather-demo
python3.11 -m venv .venv
.venv/bin/python -m pip install Pillow numpy soundfile num2words
.venv/bin/python render.py forecasts/ipma-day0.json rerender-28.mp4
.venv/bin/python render.py forecasts/ipma-day1.json rerender-29.mp4
```

The provided WAV clips are sufficient to reproduce these examples, so the voice model is not required for a re-render. Fonts use the Mac's Arial/Avenir families, with a DejaVu fallback.

## Update the forecast and voice

Install the TTS package in the same environment. Its dependency chain currently includes prereleases:

```sh
.venv/bin/python -m pip install --pre 'tts_eu_pt==0.1.1'
curl -fL 'https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day0.json' -o forecasts/ipma-day0.json
curl -fL 'https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day1.json' -o forecasts/ipma-day1.json
.venv/bin/python make_speech.py
.venv/bin/python render.py forecasts/ipma-day0.json new-day0.mp4
.venv/bin/python render.py forecasts/ipma-day1.json new-day1.mp4
```

On first voice synthesis, the model is downloaded from [logus2k/kokoro_tts_eu_pt](https://huggingface.co/logus2k/kokoro_tts_eu_pt). Subsequent synthesis is local. The renderer checks that each saved narration line matches the current input values before rendering, and fails instead of silently pairing stale speech with updated data.

This is a proof of concept rather than a production weather system: it has four fixed locations, a finite set of weather descriptions, no automatic warning handling, no publishing scheduler, and no monitoring. IPMA's update timestamp and forecast data should be reviewed before publication. The island scene shows the forecast for Ponta Delgada, not every location in the Azores.

## Local human presenter

`human_presenter.py` generates a forecast with a realistic-looking, lip-synced presenter. It uses the same Kokoro voice and the [MuseTalk 1.5 MLX port](https://github.com/xocialize/musetalk-mlx) on the Mac's GPU. The presenter image is replaceable: give `--portrait` a front-facing, well-lit image of a person whose likeness you have permission to animate. The included example is a synthetic, fictional person; see [THIRD-PARTY.md](THIRD-PARTY.md) for its source and licence.

![Preview of the synthetic human presenter](human-preview.png)

This is a talking-photo prototype: MuseTalk changes the mouth area, while the input photo supplies the rest of the face and body. It does not create realistic head motion, expressions, blinking, or a full-body performance. The four city photos are archival, not live footage, and are visibly credited. Check the result, especially the voice and lip sync, before publishing.

On an Apple Silicon Mac with Python 3.11, FFmpeg, and sufficient free disk space:

```sh
cd wxagora-weather-demo
python3.11 -m venv .venv
.venv/bin/python -m pip install --pre \
  'musetalk-mlx @ git+https://github.com/xocialize/musetalk-mlx.git@c6eb30ebd1ed4d043983209813370153de9346bf' \
  'tts_eu_pt==0.1.1' 'mediapipe==0.10.14' \
  'opencv-python<5' 'opencv-contrib-python<5' Pillow num2words
.venv/bin/hf download mlx-community/MuseTalk-1.5-q8 --local-dir .models/musetalk-q8
```

The lip-sync weights are approximately 1.64 GB; the voice model adds roughly 313 MB on its first use. They are downloaded once and are not committed to this repository. Then one command fetches today's IPMA data, writes the narration, lip-syncs the presenter and renders the video:

```sh
.venv/bin/python human_presenter.py --today \
  --model .models/musetalk-q8 \
  --portrait assets/presenter-sample.png \
  --output human-today.mp4 --save-forecast human-today.json
```

This uses no paid speech or video-generation API. The current forecast still requires an internet connection; the local AI models and renderer do not. To reproduce an existing snapshot offline, replace `--today` with `--forecast forecasts/ipma-day0.json`. The `--today` mode rejects a forecast whose date does not match the current date in Lisbon, and the renderer checks narration against the input values. A scheduler can invoke the same command, but no schedule or social-media posting is installed by this project.

By default the mouth is generated and composited at 24 fps. `--lip-fps 12` is available for a faster but less fluid render. The face crop follows MediaPipe landmarks, and the mouth blend is fitted to the detected lips. The exact image and framing matter: inspect the result with any new portrait before publishing. The 15-second sample's lip-sync and composition pass took approximately 255 seconds on an M5 MacBook Pro with 32 GB, excluding speech generation and initial downloads. The sample's credit strip was finalized locally after rendering; the script now writes the same credits directly.
