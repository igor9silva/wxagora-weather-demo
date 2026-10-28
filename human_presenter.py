#!/usr/bin/env python3
"""Render an IPMA forecast with local pt-PT speech and local MuseTalk lip sync.

Works on Apple Silicon with a local MuseTalk MLX model. A forecast can be a
saved JSON snapshot or fetched with --today. No hosted speech/video API is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from make_speech import PLACES, line_for
from render import B, F, FPS, H, W, Movie, date_label, draw_logo, draw_weather_icon, ease, fit_font, short_label, text, weather_kind

BASE = Path(__file__).resolve().parent
IPMA_URL = 'https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day0.json'


def read_forecast(args, work):
    if args.today:
        request = urllib.request.Request(IPMA_URL, headers={'User-Agent':'wxagora-weather-demo/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        expected = datetime.now(ZoneInfo('Europe/Lisbon')).date().isoformat()
        if data.get('forecastDate') != expected:
            raise ValueError(f'IPMA forecast date {data.get("forecastDate")} does not match {expected}')
        path = work / 'forecast.json'
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    else:
        path = Path(args.forecast).resolve()
        data = json.loads(path.read_text())
    if data.get('owner') != 'IPMA' or not data.get('dataUpdate'):
        raise ValueError('Expected an IPMA forecast with an update timestamp')
    rows = {int(x['globalIdLocal']):x for x in data['data']}
    for ident, name in PLACES:
        if ident not in rows or rows[ident].get('tMax') is None or rows[ident].get('idWeatherType') is None:
            raise ValueError(f'Forecast is missing temperature or weather for {name}')
    return path, data


def generate_speech(data, outdir):
    from loguru import logger
    logger.remove(); logger.add(__import__('sys').stderr,level='WARNING')
    from tts_eu_pt import SAMPLE_RATE, TTS
    print('Generating Portuguese narration locally...',flush=True)
    voice = TTS(device='cpu')
    rows = {int(x['globalIdLocal']):x for x in data['data']}
    outdir.mkdir(parents=True,exist_ok=True)
    clips=[]
    for i,(ident,name) in enumerate(PLACES):
        row=rows[ident]
        script=line_for(i,name,round(float(row['tMax'])),int(row['idWeatherType']))
        samples=voice.say(script,speed=1.16)
        filename=f'{i:02d}.wav'
        sf.write(outdir/filename,samples,SAMPLE_RATE,subtype='PCM_16')
        clips.append({'place':name,'text':script,'duration':len(samples)/SAMPLE_RATE,'file':filename})
    (outdir/'manifest.json').write_text(json.dumps(clips,ensure_ascii=False,indent=2))


def load_photo(path):
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError(f'Cannot read portrait: {path}')
    if image.ndim!=3 or image.shape[2] not in (3,4): raise ValueError('Use an RGB/RGBA presenter photo')
    source=image[:,:,:3]
    gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    found=detector.detectMultiScale(gray,scaleFactor=1.08,minNeighbors=5,minSize=(60,60))
    if len(found)!=1:
        raise ValueError(f'Expected one front-facing presenter; detected {len(found)} faces')
    x,y,w,h=map(int,found[0]); margin=int(w*.12)
    x1=max(0,x-margin); y1=max(0,y-margin); x2=min(source.shape[1],x+w+margin); y2=min(source.shape[0],y+h+margin)
    if x2-x1<80 or y2-y1<80: raise ValueError('Detected face is too small')
    if image.shape[2]==4:
        alpha=image[:,:,3]
    else:
        # Isolate an already-black studio background only when the edges really
        # are black. Other photos retain their backgrounds in a rectangular card.
        corners=np.array([source[0,0],source[0,-1],source[-1,0],source[-1,-1]])
        if float(np.max(corners))<20:
            dark=(source.max(axis=2)<22).astype(np.uint8)
            _,labels=cv2.connectedComponents(dark,connectivity=8)
            border=np.unique(np.concatenate((labels[0,:],labels[-1,:],labels[:,0],labels[:,-1])))
            back=np.isin(labels,border)&(dark>0)
            alpha=(~back).astype(np.uint8)*255
            alpha=cv2.GaussianBlur(alpha,(5,5),0)
        else:
            alpha=np.full(source.shape[:2],255,dtype=np.uint8)
    # A soft lower edge avoids a hard rectangular crop across the torso.
    fade_start=int(alpha.shape[0]*.84)
    fade=np.ones(alpha.shape[0],dtype=np.float32)
    fade[fade_start:]=np.linspace(1,0,alpha.shape[0]-fade_start,dtype=np.float32)
    alpha=(alpha.astype(np.float32)*fade[:,None]).astype(np.uint8)
    return source,alpha,(x1,y1,x2,y2)


def composite_frame(movie, t, portrait, alpha, sample_credit=False):
    if t < movie.starts[0]:
        idx=0; u=0.; heading=True
    elif t >= movie.scene_end:
        idx=len(movie.scenes)-1; u=movie.scenes[idx]['duration']; heading=False
    else:
        idx=max(i for i,start in enumerate(movie.starts) if t>=start)
        u=t-movie.starts[idx]; heading=False
    scene=movie.scenes[idx]
    im=movie.background(idx,u,ease(u/.55) if not heading else 1)
    movie.atmosphere(im,scene['code'],t,idx)

    # Large, human presenter: photo may be replaced by any suitable portrait.
    rgba=cv2.cvtColor(portrait,cv2.COLOR_BGR2RGBA)
    rgba[:,:,3]=alpha
    character=Image.fromarray(rgba,'RGBA')
    width=550; height=round(width*character.height/character.width)
    character=character.resize((width,height),Image.Resampling.BILINEAR)
    shift=int(10*math.sin(t*1.2))
    im.alpha_composite(character,(190+shift,590))

    overlay=Image.new('RGBA',(W,H)); d=ImageDraw.Draw(overlay)
    draw_logo(d)
    text(d,(44,133),date_label(movie.date),B[20],(236,246,255,255),stroke=1)
    if heading:
        text(d,(44,210),'PORTUGAL',B[72],stroke=2)
        text(d,(45,294),'A PREVISÃO',B[38],(255,215,101,255),stroke=1)
    else:
        title_font=fit_font(scene['name'],620,[72,60,50,38])
        text(d,(43,200),scene['name'],title_font,stroke=2)
        text(d,(47,295),f"{scene['temp']}°",B[126],stroke=2)
        text(d,(54,447),'MÁXIMA PREVISTA',B[20],(244,248,255,255),stroke=1)
        d.rounded_rectangle((44,501,587,585),radius=24,fill=(7,34,64,210))
        draw_weather_icon(d,weather_kind(scene['code']),89,542,t)
        label=short_label(scene['code'])
        text(d,(144,527),label,fit_font(label,415,[26,24,22,20,18]))
    for j in range(4):
        x=48+j*25
        d.ellipse((x,1122,x+9,1131),fill=(255,211,92,255) if j==idx else (255,255,255,120))
    source='FONTE: IPMA  •  RETRATO: LAMBDAWALKER (CC BY 4.0)' if sample_credit else 'FONTE: IPMA  •  DEMONSTRAÇÃO'
    text(d,(44,1160),source,B[18],(247,250,255,245),stroke=1)
    if movie.updated:
        stamp=movie.updated.replace('T',' ')[:16]
        text(d,(44,1192),f'Atualização da fonte: {stamp}',F[18],(235,244,255,230),stroke=1)
    text(d,(44,1222),'APRESENTAÇÃO E VOZ SINTÉTICAS',F[18],(235,244,255,230),stroke=1)
    if sample_credit:
        text(d,(44,1248),'creativecommons.org/licenses/by/4.0',F[18],(235,244,255,230),stroke=1)
    im.alpha_composite(overlay)
    return im.convert('RGB')


def render_video(movie, photo_path, model_path, output, work, lip_fps=12, batch_size=8):
    # Import MLX only here: preview and data validation do not need the GPU.
    import mlx.core as mx
    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline
    if FPS % lip_fps:
        raise ValueError(f'Lip FPS must divide {FPS}')
    mx.set_default_device(mx.gpu)
    source,alpha,(x1,y1,x2,y2)=load_photo(photo_path)
    sample_credit=photo_path.resolve()==(BASE/'assets/presenter-sample.png').resolve()
    crop=cv2.resize(source[y1:y2,x1:x2],(256,256),interpolation=cv2.INTER_LANCZOS4)
    speech=work/'speech.wav'; mixed=work/'mixed.wav'
    movie.write_audio(speech,include_music=False)
    movie.write_audio(mixed)
    print('Loading local MuseTalk model...',flush=True)
    pipe=MuseTalkPipeline.from_pretrained_mlx(model_path)
    latent=pipe.get_latents_for_unet(crop).astype(mx.float16)
    chunks=pipe.encode_audio_from_wav(speech,fps=lip_fps).astype(mx.float16)
    n=int(chunks.shape[0]); subframes=FPS//lip_fps
    print(f'Animating {n} frames at {lip_fps} fps, compositing at {FPS} fps',flush=True)

    mask=np.zeros((256,256),np.uint8)
    # Keep the generated area tightly around the mouth; retaining original
    # cheeks and jaw avoids a soft rectangular lower-face seam.
    cv2.ellipse(mask,(128,190),(52,30),0,0,360,255,-1)
    mask=cv2.GaussianBlur(mask,(21,21),0).astype(np.float32)/255
    mask=cv2.resize(mask,(x2-x1,y2-y1))[...,None]
    active=[]
    for i in range(n):
        t=i/lip_fps
        active.append(any(start+.07<=t<=start+.18+float(s['clip']['duration'])+.10
                          for start,s in zip(movie.starts,movie.scenes)))
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-i',str(mixed),'-c:v','libx264',
         '-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k',
         '-movflags','+faststart','-shortest',str(output)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    started=time.monotonic(); generated=0
    try:
        for start in range(0,n,batch_size):
            end=min(start+batch_size,n)
            chosen=[i for i in range(start,end) if active[i]]
            generated_faces={}
            if chosen:
                # Keep the batch shape stable, avoiding repeated GPU compilation.
                padded=chosen+[chosen[-1]]*(batch_size-len(chosen))
                latents=mx.repeat(latent,batch_size,axis=0)
                audio_chunks=mx.stack([chunks[i] for i in padded],axis=0)
                results=pipe.generate_faces(latents,audio_chunks)
                generated_faces=dict(zip(chosen,results[:len(chosen)]))
                generated+=len(chosen)
            for i in range(start,end):
                frame=source.copy()
                if i in generated_faces:
                    face=cv2.resize(generated_faces[i],(x2-x1,y2-y1),interpolation=cv2.INTER_LANCZOS4)
                    base=frame[y1:y2,x1:x2]
                    frame[y1:y2,x1:x2]=np.clip(face*mask+base*(1-mask),0,255).astype(np.uint8)
                for step in range(subframes):
                    t=(i*subframes+step)/FPS
                    if t<movie.total: proc.stdin.write(composite_frame(movie,t,frame,alpha,sample_credit).tobytes())
            if start%48==0:
                print(f'Lip frames {end}/{n} ({time.monotonic()-started:.0f}s)',flush=True)
    finally:
        proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
    print(f'Wrote {output}; generated {generated} lip frames in {time.monotonic()-started:.1f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--forecast',help='Saved IPMA daily JSON snapshot')
    source.add_argument('--today',action='store_true',help='Fetch and validate the current IPMA forecast')
    p.add_argument('--portrait',type=Path,default=BASE/'assets/presenter-sample.png')
    p.add_argument('--model',type=Path,help='Directory with the downloaded MuseTalk MLX weights')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-forecast',type=Path,help='Save a copy of the forecast used for this render')
    p.add_argument('--audio-dir',type=Path,help='Reuse existing matching speech clips instead of generating them')
    p.add_argument('--preview',type=float,help='Save a single PNG frame; requires --audio-dir')
    p.add_argument('--lip-fps',type=int,default=12,choices=[12,24])
    args=p.parse_args()
    if args.preview is None and not args.model: p.error('--model is required for a video render')
    if args.preview is not None and not args.audio_dir: p.error('--preview requires --audio-dir')
    if not args.portrait.is_file(): p.error('the presenter portrait does not exist')
    if args.preview is None:
        for name in ('config.json','unet.safetensors','vae.safetensors','whisper_encoder.safetensors'):
            if not (args.model/name).is_file(): p.error(f'the local model is missing {name}')
    with tempfile.TemporaryDirectory(prefix='wx-presenter-') as tmp:
        work=Path(tmp); forecast,data=read_forecast(args,work)
        if args.save_forecast:
            args.save_forecast.parent.mkdir(parents=True,exist_ok=True)
            args.save_forecast.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        audio_dir=args.audio_dir.resolve() if args.audio_dir else work/'audio'
        if not args.audio_dir: generate_speech(data,audio_dir)
        movie=Movie(forecast,audio_dir=audio_dir)
        if args.preview is not None:
            source,alpha,_=load_photo(args.portrait)
            sample_credit=args.portrait.resolve()==(BASE/'assets/presenter-sample.png').resolve()
            composite_frame(movie,args.preview,source,alpha,sample_credit).save(args.output)
            print(f'Wrote preview {args.output}',flush=True)
        else:
            started=time.monotonic()
            render_video(movie,args.portrait,args.model.resolve(),args.output.resolve(),work,lip_fps=args.lip_fps)
            print(f'Total render time {time.monotonic()-started:.1f}s (excludes speech generation)',flush=True)


if __name__=='__main__':
    main()
