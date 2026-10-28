#!/usr/bin/env python3
"""Prepare date- and forecast-specific image prompts and local Portuguese speech.

The image prompts are intended for Codex's built-in image-generation tool;
this program does not silently substitute an API or call a paid service.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from num2words import num2words

IPMA = 'https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day0.json'
PLACES = [
    (1131200, 'Porto', 'porto', "Porto's Ribeira and the Dom Luís I bridge over the Douro, viewed from Vila Nova de Gaia"),
    (1110600, 'Lisboa', 'lisboa', "Lisbon's Praça do Comércio and Arco da Rua Augusta, with its yellow Pombaline buildings"),
    (1080500, 'Faro', 'faro', 'the Marina de Faro in Faro city, Algarve, with its boats, palms and waterfront'),
    (3420300, 'Ponta Delgada', 'ponta-delgada', 'the black-and-white Portas da Cidade in Ponta Delgada, São Miguel, Azores'),
]

# IPMA weather codes: a visual description and a natural Portuguese phrase.
WEATHER = {
    0: ('unspecified weather, neutral light', None),
    1: ('clear sunny sky', 'céu limpo'),
    2: ('sunny intervals and a few clouds', 'poucas nuvens'),
    3: ('broken clouds with some sun', 'períodos de nuvens'),
    4: ('heavily overcast sky without rain', 'céu muito nublado'),
    5: ('high thin clouds', 'nuvens altas'),
    6: ('cloudy sky with rain showers and wet pavement', 'aguaceiros'),
    7: ('cloudy sky with light rain and wet pavement', 'chuva fraca'),
    8: ('heavy rain showers and wet pavement', 'aguaceiros fortes'),
    9: ('cloudy sky with rain showers and wet pavement', 'chuva ou aguaceiros'),
    10: ('cloudy sky with drizzle and wet pavement', 'chuviscos'),
    11: ('cloudy sky with heavy rain and wet pavement', 'chuva forte'),
    12: ('cloudy sky with periods of rain and wet pavement', 'períodos de chuva'),
    13: ('cloudy sky with light rain and wet pavement', 'chuva fraca'),
    14: ('cloudy sky with heavy rain and wet pavement', 'chuva forte'),
    15: ('cloudy sky with drizzle and wet pavement', 'chuviscos'),
    16: ('misty atmosphere', 'neblina'),
    17: ('dense fog', 'nevoeiro'),
    18: ('snow falling', 'neve'),
    19: ('storm clouds and lightning in the distance', 'trovoada'),
    20: ('rain showers and storm clouds, with a possibility of thunder', 'aguaceiros e possibilidade de trovoada'),
    21: ('storm clouds and hail', 'granizo'),
    22: ('frost on the ground', 'geada'),
    23: ('rain and storm clouds, with a possibility of thunder', 'chuva e possibilidade de trovoada'),
    24: ('large convective clouds', 'nuvens convectivas'),
    25: ('periods of heavy cloud', 'períodos de muito nublado'),
    26: ('fog', 'nevoeiro'),
    27: ('overcast sky', 'céu nublado'),
    28: ('snow showers', 'aguaceiros de neve'),
    29: ('rain and snow', 'chuva e neve'),
    30: ('rain and snow', 'chuva e neve'),
}


def make_plan(forecast):
    if forecast.get('owner') != 'IPMA' or not forecast.get('forecastDate') or not forecast.get('dataUpdate'):
        raise ValueError('Expected a dated IPMA forecast with an update timestamp')
    rows = {int(row['globalIdLocal']): row for row in forecast['data']}
    scenes = []
    for ident, name, slug, landmark in PLACES:
        if ident not in rows:
            raise ValueError(f'Missing IPMA forecast for {name}')
        row = rows[ident]
        code = int(row['idWeatherType'])
        if code not in WEATHER:
            raise ValueError(f'Unknown weather code for {name}: {code}')
        if code == 0:
            raise ValueError(f'No weather information available for {name}')
        visual, spoken = WEATHER[code]
        temp = round(float(row['tMax']))
        degrees = num2words(temp, lang='pt')
        sentence = f'{name}, ' + (f'{spoken} e ' if spoken else '') + f'máxima de {degrees} graus.'
        prompt = (
            'Use case: photorealistic-natural. Create a portrait-format 9:16 background plate '
            f'for a Portuguese television weather forecast. A recognizable, geographically plausible view of {landmark}. '
            f"The IPMA daily forecast for {forecast['forecastDate']} is {visual}, with a maximum of {temp}°C. "
            'Natural realistic color and camera perspective. No person in the foreground, no map, '
            'no lettering, numbers, captions, graphics, logos, watermark or fake weather symbols. '
            'Leave usable open space in the lower half for a full-body presenter.'
        )
        scenes.append({'name': name, 'slug': slug, 'location_id': ident, 'temperature': temp,
                       'weather_code': code, 'spoken_condition': spoken, 'text': sentence,
                       'image_file': f'{slug}.png', 'image_prompt': prompt})
    return {'date': forecast['forecastDate'], 'source': IPMA, 'source_updated': forecast['dataUpdate'], 'scenes': scenes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--today', action='store_true', help='Fetch and validate the current Lisbon date')
    source.add_argument('--forecast', type=Path, help='Use a saved IPMA forecast snapshot')
    p.add_argument('--out', type=Path, required=True, help='Directory for plan, snapshot and optional voice clips')
    p.add_argument('--speech', action='store_true', help='Generate the narration locally with Kokoro pt-PT')
    args = p.parse_args()
    if args.today:
        request = urllib.request.Request(IPMA, headers={'User-Agent': 'WXAgoraProof/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            forecast = json.load(response)
        today = datetime.now(ZoneInfo('Europe/Lisbon')).date().isoformat()
        if forecast.get('forecastDate') != today:
            raise ValueError(f'IPMA forecast date {forecast.get("forecastDate")} does not match {today}')
    else:
        forecast = json.loads(args.forecast.read_text())
    plan = make_plan(forecast)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'forecast.json').write_text(json.dumps(forecast, ensure_ascii=False, indent=2) + '\n')
    if args.speech:
        import soundfile as sf
        from loguru import logger
        logger.remove()
        from tts_eu_pt import SAMPLE_RATE, TTS
        voice = TTS(device='cpu')
        for scene in plan['scenes']:
            audio = voice.say(scene['text'], speed=1.05)
            scene['audio_file'] = f"{scene['slug']}.wav"
            scene['audio_duration'] = len(audio) / SAMPLE_RATE
            sf.write(args.out / scene['audio_file'], audio, SAMPLE_RATE, subtype='PCM_16')
            print(scene['name'], f"{scene['audio_duration']:.2f}s", scene['text'], flush=True)
    (args.out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {args.out / "plan.json"}; generate the four image prompts with the built-in image tool.')


if __name__ == '__main__':
    main()
