import copy
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_daily import make_plan
from render_fullbody import FPS, VOICE_RATE, assemble, frame_count, validate_images

BASE = Path(__file__).resolve().parent.parent


class FullBodyChecks(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan(json.loads((BASE / 'forecasts' / 'ipma-human-2026-10-28.json').read_text()))
        self.images = BASE / 'assets' / 'generated' / '2026-10-28'

    def test_weather_changes_invalidate_generated_image(self):
        validate_images(self.plan, self.images)
        changed = copy.deepcopy(self.plan)
        changed['scenes'][0]['weather_code'] = 1
        with self.assertRaisesRegex(ValueError, 'fresh, matching image'):
            validate_images(changed, self.images)

    def test_image_filename_must_match_registered_forecast(self):
        changed = copy.deepcopy(self.plan)
        changed['scenes'][0]['image_file'] = 'lisboa.png'
        with self.assertRaisesRegex(ValueError, 'fresh, matching image'):
            validate_images(changed, self.images)

    def test_animation_duration_does_not_cut_speech(self):
        for seconds in (2.0, 4.275, 4.75, 8.0):
            frames = frame_count(seconds)
            self.assertEqual((frames - 1) % 8, 0)
            self.assertGreater(frames / FPS, seconds)

    def test_unrecognized_weather_is_not_invented(self):
        raw = json.loads((BASE / 'forecasts' / 'ipma-human-2026-10-28.json').read_text())
        next(row for row in raw['data'] if int(row['globalIdLocal']) == 1131200)['idWeatherType'] = 999
        with self.assertRaisesRegex(ValueError, 'Unknown weather code'):
            make_plan(raw)

    def test_missing_weather_is_not_invented(self):
        raw = json.loads((BASE / 'forecasts' / 'ipma-human-2026-10-28.json').read_text())
        next(row for row in raw['data'] if int(row['globalIdLocal']) == 1131200)['idWeatherType'] = 0
        with self.assertRaisesRegex(ValueError, 'No weather information'):
            make_plan(raw)

    def test_assembled_speech_does_not_drift_at_city_cuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes, segments, waves = [], [], []
            for i, seconds in enumerate((.47, .57, .61)):
                t = np.arange(round(seconds * VOICE_RATE)) / VOICE_RATE
                wave = (.35 * np.sin(2 * np.pi * ((210 + 90*i) * t + 350 * t*t))).astype('float32')
                audio = root / f'{i}.wav'
                sf.write(audio, wave, VOICE_RATE)
                duration = len(wave) / VOICE_RATE
                scene_duration = math.ceil(duration * FPS) / FPS
                segment = root / f'{i}.mp4'
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                                '-i', 'color=blue:s=64x64:r=24', '-i', str(audio), '-af', 'apad',
                                '-t', str(scene_duration), '-c:v', 'libx264', '-c:a', 'aac', str(segment)],
                               check=True)
                scenes.append({'audio_file': audio.name, 'audio_duration': duration})
                segments.append(segment)
                waves.append(wave)
            output = root / 'joined.mp4'
            assemble(segments, scenes, root, output)
            decoded = root / 'joined.wav'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(output),
                            '-vn', '-ar', str(VOICE_RATE), '-ac', '1', str(decoded)], check=True)
            joined, rate = sf.read(decoded)
            self.assertEqual(rate, VOICE_RATE)
            expected = 0
            for scene, wave in zip(scenes, waves):
                start = max(0, expected - round(.06 * rate))
                end = min(len(joined), expected + len(wave) + round(.06 * rate))
                step = 6
                correlation = np.correlate(joined[start:end:step], wave[::step], mode='valid')
                actual = start + int(np.argmax(correlation)) * step
                self.assertLess(abs(actual - expected) / rate, .01)
                expected += math.ceil(scene['audio_duration'] * FPS) * rate // FPS


if __name__ == '__main__':
    unittest.main()
