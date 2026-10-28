#!/usr/bin/env python3
"""Make a reusable set of local pt-PT voice clips from saved IPMA snapshots."""
import json
import os
import sys
from pathlib import Path

import soundfile as sf

BASE = Path(__file__).resolve().parent
FORECASTS = BASE / "forecasts"
OUT = BASE / "audio"
PLACES = [(1131200, "Porto"), (1110600, "Lisboa"), (1080500, "Faro"), (3420300, "Ponta Delgada")]


def condition_phrase(code):
    if code in (6, 7, 9, 10, 12, 13, 15):
        return "com chuva"
    if code in (8, 11, 14):
        return "com chuva forte"
    if code in (19, 20, 23):
        return "com possibilidade de trovoada"
    if code in (18, 28, 29, 30):
        return "com neve"
    if code in (16, 17, 26):
        return "com nevoeiro"
    return None


def line_for(i, name, maximum, code):
    if i == 0:
        return f"Porto, máxima de {maximum} graus."
    if i == 1:
        return f"Lisboa, {maximum} graus."
    if i == 2:
        return f"Faro, {maximum} graus."
    weather = condition_phrase(code)
    from num2words import num2words
    degrees = num2words(maximum, lang="pt")
    if weather in ("com chuva", "com chuva forte"):
        ending = " e " + weather.removeprefix("com ") + "."
    else:
        ending = f", {weather}." if weather else "."
    return f"Em São Miguel, Ponta Delgada tem {degrees} graus" + ending


def main():
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from tts_eu_pt import TTS, SAMPLE_RATE
    synthesizer = TTS(device="cpu")
    for path in sorted(FORECASTS.glob("ipma-day[01].json")):
        raw = json.loads(path.read_text())
        by_id = {int(item["globalIdLocal"]): item for item in raw["data"]}
        output_dir = OUT / raw["forecastDate"]
        output_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for i, (place_id, name) in enumerate(PLACES):
            data = by_id[place_id]
            text = line_for(i, name, round(float(data["tMax"])), int(data["idWeatherType"]))
            wav = synthesizer.say(text, speed=1.16)
            dest = output_dir / f"{i:02d}.wav"
            sf.write(dest, wav, SAMPLE_RATE, subtype="PCM_16")
            result.append({"place": name, "text": text, "duration": len(wav) / SAMPLE_RATE, "file": dest.name})
            print(raw["forecastDate"], name, f"{len(wav)/SAMPLE_RATE:.2f}s", text, flush=True)
        (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
