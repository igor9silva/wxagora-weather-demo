import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from human_presenter import BASE, load_photo, read_forecast
from render import Movie


class PresenterChecks(unittest.TestCase):
    def setUp(self):
        self.forecast = BASE / 'forecasts' / 'ipma-day0.json'
        self.audio = BASE / 'audio' / '2026-10-28'

    def test_rejects_wrong_daily_date(self):
        data = json.loads(self.forecast.read_text())
        data['forecastDate'] = '2001-01-01'
        response = io.BytesIO(json.dumps(data).encode())
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                read_forecast(Namespace(today=True, forecast=None), Path(tmp))

    def test_rejects_stale_narration(self):
        data = json.loads(self.forecast.read_text())
        for item in data['data']:
            if item['globalIdLocal'] == 1131200:
                item['tMax'] += 1
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / 'forecast.json'
            altered.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Stale narration'):
                Movie(altered, audio_dir=self.audio)

    def test_presenter_has_one_face_and_transparent_background(self):
        image, alpha, box = load_photo(BASE / 'assets' / 'presenter-sample.png')
        self.assertEqual(image.shape[:2], alpha.shape)
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertGreater(int(alpha[180, 256]), 200)
        self.assertGreater(box[2] - box[0], 100)

    def test_speech_track_has_silent_intro_and_audible_speech(self):
        movie = Movie(self.forecast, audio_dir=self.audio)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'speech.wav'
            movie.write_audio(path, include_music=False)
            audio, sr = sf.read(path)
            self.assertEqual(sr, 24000)
            self.assertLess(np.max(np.abs(audio[:int(.4*sr)])), .001)
            self.assertGreater(np.max(np.abs(audio[int(1.0*sr):int(3.0*sr)])), .1)


if __name__ == '__main__':
    unittest.main()
