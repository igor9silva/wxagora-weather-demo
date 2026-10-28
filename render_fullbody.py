#!/usr/bin/env python3
"""Animate a full-body reporter locally and compose forecast-specific scenes.

The input photos are generated per forecast with Codex's image tool. The local
audio-to-video model generates whole frames, conditioned on the narration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
import soundfile as sf

from render import B, F, H, W, date_label, font, short_label, text

FPS = 24
VOICE_RATE = 24000
MOTION_PROMPT = (
    'A full body Portuguese television weather reporter stands against a uniform bright green studio backdrop. '
    'Facing the camera, she speaks European Portuguese into her handheld microphone and moves naturally: '
    'she blinks, shifts her weight, turns her shoulders slightly and makes clear fluid gestures with her free hand. '
    'Her microphone, normal hands, feet and entire body remain visible. Photorealistic, fixed camera, '
    'steady shot, one person, no cuts, no on-screen text.'
)
IMAGE_STRENGTH = .85


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_images(plan, directory):
    meta = json.loads((directory / 'manifest.json').read_text())
    if meta.get('date') != plan['date']:
        raise ValueError('Generated image set is for another date')
    indexed = {row['slug']: row for row in meta['scenes']}
    for scene in plan['scenes']:
        row = indexed.get(scene['slug'])
        if (not row or row['weather_code'] != scene['weather_code'] or
                row['temperature'] != scene['temperature'] or row['file'] != scene['image_file']):
            raise ValueError(f"Generate a fresh, matching image for {scene['name']}")
        image = directory / row['file']
        if not image.is_file() or hash_file(image) != row['sha256']:
            raise ValueError(f"Missing or changed image for {scene['name']}")


def frame_count(seconds):
    # LTX-2 accepts 8k+1 frames; leave a little room after the narration.
    return max(49, math.ceil(((seconds + .15) * FPS - 1) / 8) * 8 + 1)


def make_graphic(plan, scene, path):
    overlay = Image.new('RGBA', (W, H))
    d = ImageDraw.Draw(overlay)
    # A shallow header leaves the presenter's face, body and city visible.
    d.rounded_rectangle((16, 14, 280, 91), radius=10, fill=(5, 22, 37, 205))
    text(d, (28, 17), scene['name'].upper(), B[26])
    condition = {20: 'AGUACEIROS E POSS. TROVOADA', 23: 'CHUVA E POSS. TROVOADA',
                 25: 'PERÍODOS MUITO NUBLADOS'}.get(
        scene['weather_code'], short_label(scene['weather_code']))
    condition_font = next((f for f in (F[18], font(16), font(14)) if d.textlength(condition, font=f) <= 240), font(12))
    text(d, (28, 58), condition, condition_font, (226, 239, 247, 255))
    text(d, (360, 20), 'WXAGORA', B[18], anchor='ma', stroke=2)
    d.rounded_rectangle((490, 14, 704, 91), radius=10, fill=(5, 22, 37, 205))
    text(d, (694, 17), date_label(plan['date']), font(16), (242, 248, 252, 255), anchor='ra')
    text(d, (694, 42), f"{scene['temperature']}°", B[38], anchor='ra')
    text(d, (500, 68), 'MÁX. · IPMA · IA', font(13), (218, 234, 244, 255))
    overlay.save(path)


def composite(plan, scene, images, clip, output, work):
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=duration', '-of', 'json', str(clip)]))
    duration = float(info['streams'][0]['duration'])
    if duration + .01 < float(scene['audio_duration']):
        raise ValueError(f"Animation for {scene['name']} is shorter than its narration")
    # The model requires extra frame-grid padding, but it can fade its final
    # padded frames. Keep the entire source narration and only trim that tail.
    duration = math.ceil(float(scene['audio_duration']) * FPS) / FPS
    graphic = work / f"{scene['slug']}-graphics.png"
    make_graphic(plan, scene, graphic)
    audio = work.parent / scene['audio_file']
    if not audio.is_file():
        raise FileNotFoundError(audio)
    # Chroma key isolates the entire generated body. This is not face inpainting.
    filters = (
        '[0:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1[bg];'
        '[1:v]setpts=PTS-STARTPTS,scale=720:1280:force_original_aspect_ratio=decrease,'
        'format=yuva420p,chromakey=0x3cd64c:0.09:0.03,despill=type=green:mix=0.5[person];'
        '[bg][person]overlay=(W-w)/2:(H-h)/2+25:shortest=1[scene];'
        '[scene][3:v]overlay=0:0:shortest=1,format=yuv420p[v]'
    )
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-loop', '1', '-framerate', str(FPS), '-i', str(images / scene['image_file']),
        '-i', str(clip), '-i', str(audio), '-loop', '1', '-framerate', str(FPS), '-i', str(graphic),
        '-filter_complex', filters, '-map', '[v]', '-map', '2:a:0', '-af', 'apad', '-t', str(duration),
        '-r', str(FPS), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-c:a', 'aac',
        '-b:a', '160k', '-movflags', '+faststart', str(output),
    ], check=True)


def assemble(segments, scenes, audio_dir, output):
    """Join pictures and source audio on the same frame/sample grid.

    Copy-concatenating AAC-encoded segments introduces encoder padding at
    each cut, progressively delaying speech relative to the presenter.
    """
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
    for segment in segments:
        command.extend(['-i', str(segment)])
    for scene in scenes:
        command.extend(['-i', str(audio_dir / scene['audio_file'])])
    filters = []
    for i, scene in enumerate(scenes):
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS[v{i}]')
        samples = math.ceil(float(scene['audio_duration']) * FPS) * VOICE_RATE // FPS
        filters.append(f'[{i + len(scenes)}:a]aresample={VOICE_RATE},aformat=channel_layouts=mono,'
                       f'apad,atrim=end_sample={samples},asetpts=PTS-STARTPTS[a{i}]')
    filters.append(''.join(f'[v{i}]' for i in range(len(scenes))) +
                   f'concat=n={len(scenes)}:v=1:a=0[v]')
    filters.append(''.join(f'[a{i}]' for i in range(len(scenes))) +
                   f'concat=n={len(scenes)}:v=0:a=1[a]')
    command.extend(['-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', str(output)])
    subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--presenter', type=Path, required=True)
    p.add_argument('--ltx', type=Path, required=True, help='Path to the installed ltx-2-mlx executable')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gemma', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', help='Render only the scene with this slug, for a short test')
    p.add_argument('--reuse-clips', action='store_true', help='Use existing clips in the plan directory')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_images(plan, args.images)
    scenes = [s for s in plan['scenes'] if not args.only or s['slug'] == args.only]
    if not scenes:
        p.error('unknown scene slug')
    work = args.plan.parent / 'presenter-work'
    work.mkdir(parents=True, exist_ok=True)
    composited = []
    for scene in scenes:
        if not scene.get('audio_file') or not scene.get('audio_duration'):
            raise ValueError('Run prepare_daily.py with --speech first')
        clip = work / f"{scene['slug']}-presenter.mp4"
        clip_manifest = work / f"{scene['slug']}-presenter.json"
        audio = args.plan.parent / scene['audio_file']
        if abs(sf.info(str(audio)).duration - float(scene['audio_duration'])) > .01:
            raise ValueError(f"Narration duration for {scene['name']} differs from its plan; regenerate the plan")
        prompt = MOTION_PROMPT + f" She says: {scene['text']}"
        signature = {'audio_sha256': hash_file(audio), 'presenter_sha256': hash_file(args.presenter),
                     'prompt': prompt, 'image_strength': IMAGE_STRENGTH,
                     'frames': frame_count(scene['audio_duration'])}
        if not args.reuse_clips:
            # The audio-to-video model requires a source track at least as
            # long as its 8k+1-frame output. Pad silence, never trim speech.
            padded = work / f"{scene['slug']}-padded.wav"
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(audio),
                            '-af', 'apad', '-t', str(frame_count(scene['audio_duration']) / FPS + .04),
                            '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', str(padded)], check=True)
            command = [str(args.ltx.resolve()), 'a2v', '--distilled', '--low-ram',
                       '--model', str(args.model.resolve()), '--gemma', str(args.gemma.resolve()),
                       '--prompt', prompt, '--audio', str(padded.resolve()),
                       '--image', str(args.presenter.resolve()), '0', str(IMAGE_STRENGTH), '--height', '768', '--width', '448',
                       '--frames', str(frame_count(scene['audio_duration'])), '--frame-rate', str(FPS),
                       '--seed', '42', '--output', str(clip.resolve())]
            print(f"Generating full-body video: {scene['name']} ({frame_count(scene['audio_duration'])} frames)", flush=True)
            subprocess.run(command, check=True)
            clip_manifest.write_text(json.dumps({**signature, 'clip_sha256': hash_file(clip)}, indent=2) + '\n')
        if not clip.is_file():
            raise FileNotFoundError(clip)
        if not clip_manifest.is_file():
            raise ValueError(f"Missing provenance for {clip}; regenerate it")
        saved = json.loads(clip_manifest.read_text())
        if any(saved.get(key) != value for key, value in signature.items()) or saved.get('clip_sha256') != hash_file(clip):
            raise ValueError(f"Stale or modified presenter clip for {scene['name']}; regenerate it")
        segment = work / f"{scene['slug']}-scene.mp4"
        composite(plan, scene, args.images, clip, segment, work)
        composited.append(segment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if len(composited) == 1:
        import shutil
        shutil.copyfile(composited[0], args.output)
    else:
        assemble(composited, scenes, args.plan.parent, args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
