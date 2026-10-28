# Portugal em viagem — WXAgora concept

Two independently made, non-official vertical weather videos from one reusable visual system. The images show a **stylized fictional miniature**, not an accurate map or live camera view. The mascot is a moving 2D cutout over the artwork, not a rigged, lip-synced 3D character.

![Preview of the weather video](preview.png)

| Video | Forecast date | Duration |
| --- | --- | --- |
| [28 October video](01-28-outubro.mp4) | 28 October 2026 | 15.3 s |
| [29 October video](02-29-outubro.mp4) | 29 October 2026 | 15.5 s |

The dates and weather data are saved snapshots, not a live forecast. City names, daily maximum temperatures and weather descriptions are read from `forecasts/ipma-day0.json` and `forecasts/ipma-day1.json`. Both snapshots report IPMA's source update timestamp `2026-10-28T11:31:03`. The JSON is formatted for readability, and the city and weather-type reference tables are included as well.

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
