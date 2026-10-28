#!/usr/bin/env python3
"""Register images actually generated for a plan, binding them to that forecast."""
import argparse
import json
from pathlib import Path

from render_fullbody import hash_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--actual-prompts', type=Path, help='Optional JSON object of slug to the exact prompt used')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    prompts = json.loads(args.actual_prompts.read_text()) if args.actual_prompts else {}
    scenes = []
    for scene in plan['scenes']:
        image = args.images / scene['image_file']
        if not image.is_file():
            raise FileNotFoundError(image)
        scenes.append({'slug': scene['slug'], 'file': scene['image_file'], 'temperature': scene['temperature'],
                       'weather_code': scene['weather_code'], 'sha256': hash_file(image),
                       'prompt': prompts.get(scene['slug'], scene['image_prompt'])})
    meta = {'date': plan['date'], 'source_updated': plan['source_updated'],
            'generator': 'Codex built-in image generation', 'scenes': scenes}
    path = args.images / 'manifest.json'
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
