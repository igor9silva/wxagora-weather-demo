# Portugal em viagem — WXAgora concept

Independent, non-official vertical weather videos from one reusable visual system. The backgrounds show a **stylized fictional miniature**, not an accurate map or live camera view. The original examples use a moving 2D mascot; the newer example uses a synthetic human presenter with locally generated lip motion.

![Preview of the weather video](preview.png)

| Video | Forecast date | Duration |
| --- | --- | --- |
| [28 October video](01-28-outubro.mp4) | 28 October 2026 | 15.3 s |
| [29 October video](02-29-outubro.mp4) | 29 October 2026 | 15.5 s |
| [Human presenter, 28 October](03-human-presenter-2026-10-28.mp4) | 28 October 2026 | 15.1 s |

The dates and weather data are saved snapshots, not a live forecast. City names, daily maximum temperatures and weather descriptions are read from `forecasts/ipma-day0.json` and `forecasts/ipma-day1.json`. Both snapshots report IPMA's source update timestamp `2026-10-28T11:31:03`. The JSON is formatted for readability, and the city and weather-type reference tables are included as well.

The human-presenter video uses the separately saved [12:31 snapshot](forecasts/ipma-human-2026-10-28.json).

Source: [IPMA API](https://api.ipma.pt/). The API's terms require attribution and limit free reuse to non-profit purposes. Check the current terms and appropriate licence before commercial use. This demonstration is not affiliated with WXAgora or IPMA.

## How it was made

- Portrait video: 720 × 1280, 24 fps, H.264/AAC, rendered locally with Python, Pillow and FFmpeg.
- Voice: `tts_eu_pt` 0.1.1, a Kokoro-82M model fine-tuned for European Portuguese, running locally on the Mac CPU. The generated clips and their exact text are in `audio/`. No macOS system voice was used in the videos.
- The short background music and transition sounds are synthesized by `render.py` from mathematical waveforms. No music catalogue or paid audio service was used.
- The landscape and mascot are one-time bitmap assets generated using Codex's built-in image-generation tool. Their actual prompts are in `ART-PROMPTS.md`. They are animated and composited locally; no video-generation API was used.

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

This is a talking-photo prototype: MuseTalk changes the mouth area, while the input photo supplies the rest of the face and body. It does not create realistic head motion, expressions, blinking, or a full-body performance, and the mouth can look soft. The background is the existing stylized illustration. Check the result, especially the voice and lip sync, before publishing.

On an Apple Silicon Mac with Python 3.11, FFmpeg, and sufficient free disk space:

```sh
cd wxagora-weather-demo
python3.11 -m venv .venv
.venv/bin/python -m pip install --pre \
  'musetalk-mlx @ git+https://github.com/xocialize/musetalk-mlx.git@c6eb30ebd1ed4d043983209813370153de9346bf' \
  'tts_eu_pt==0.1.1' 'opencv-python<5' Pillow num2words
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

By default the mouth is generated at 12 fps and composited into a 24 fps video. Add `--lip-fps 24` for smoother motion at a longer render time. On the test MacBook Pro (M5, 32 GB), the 12-fps lip-sync and composition pass for a 15-second video took about 158 seconds; the 24-fps pass took about 241 seconds. These measurements exclude speech generation and model downloads. The published example received a further local mouth-blend and credit adjustment; the current script includes those adjustments for future renders.
