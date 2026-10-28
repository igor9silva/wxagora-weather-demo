#!/usr/bin/env python3
"""Render an IPMA forecast with local pt-PT speech and local MuseTalk lip sync.

Works on Apple Silicon with a local MuseTalk MLX model. A forecast can be a
saved JSON snapshot or fetched with --today. No hosted speech/video API is used.
"""
from __future__ import annotations

import argparse
import json
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
from PIL import Image, ImageDraw, ImageOps

from make_speech import PLACES, line_for
from render import B, F, FPS, H, W, Movie, date_label, draw_logo, draw_weather_icon, short_label, text, weather_kind

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
    import mediapipe as mp

    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError(f'Cannot read portrait: {path}')
    if image.ndim!=3 or image.shape[2] not in (3,4): raise ValueError('Use an RGB/RGBA presenter photo')
    source=image[:,:,:3]
    # MuseTalk expects a landmark-aligned crop with the nose bridge halfway
    # down and the chin at the bottom, not a generic face-detector square.
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=2,
                                         refine_landmarks=False,min_detection_confidence=.5) as mesh:
        found=mesh.process(cv2.cvtColor(source,cv2.COLOR_BGR2RGB)).multi_face_landmarks or []
    if len(found)!=1:
        raise ValueError(f'Expected one front-facing presenter; detected {len(found)} faces')
    pts=np.array([[p.x*source.shape[1],p.y*source.shape[0]] for p in found[0].landmark])
    middle=int(pts[6,1]); y2=min(source.shape[0],int(pts[:,1].max()))
    x1=max(0,int(pts[:,0].min())); x2=min(source.shape[1],int(pts[:,0].max()))
    y1=max(0,2*middle-y2)
    if x2-x1<80 or y2-y1<80: raise ValueError('Detected face is too small')
    mouth_x=(pts[61,0]+pts[291,0])/2
    mouth_y=(pts[0,1]+pts[17,1])/2
    mouth_width=float(np.linalg.norm(pts[61]-pts[291]))
    if mouth_width<12 or not (x1<mouth_x<x2 and middle<mouth_y<y2):
        raise ValueError('Could not find a suitable front-facing mouth')
    mask=np.zeros((y2-y1,x2-x1),np.uint8)
    rx=max(1,round(mouth_width*.70)); ry=max(1,round(mouth_width*.48))
    cv2.ellipse(mask,(round(mouth_x-x1),round(mouth_y-y1)),(rx,ry),0,0,360,255,-1)
    blur=max(3,int(mouth_width*.20)//2*2+1)
    mask=cv2.GaussianBlur(mask,(blur,blur),0).astype(np.float32)[...,None]/255
    # Never cut a blend at the boundary of the model's crop.
    feather=min(max(2,blur//2),mask.shape[0]//4,mask.shape[1]//4)
    ramp=np.linspace(0,1,feather,dtype=np.float32)
    mask[:feather,:,:]*=ramp[:,None,None]; mask[-feather:,:,:]*=ramp[::-1,None,None]
    mask[:,:feather,:]*=ramp[None,:,None]; mask[:,-feather:,:]*=ramp[None,::-1,None]
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
    return source,alpha,(x1,y1,x2,y2),mask


def load_city_assets():
    directory=BASE/'assets'/'cities'
    credits=json.loads((directory/'credits.json').read_text())
    if [item['city'] for item in credits] != [name.upper() for _,name in PLACES]:
        raise ValueError('City photo metadata does not match forecast locations')
    images=[]
    for entry in credits:
        with Image.open(directory/entry['file']) as source:
            images.append(ImageOps.fit(source.convert('RGB'),(W,480),Image.Resampling.LANCZOS).convert('RGBA'))
    return credits,images


def composite_frame(movie, t, portrait, alpha, cities, sample_credit=False):
    if t < movie.starts[0]:
        idx=0
    elif t >= movie.scene_end:
        idx=len(movie.scenes)-1
    else:
        idx=max(i for i,start in enumerate(movie.starts) if t>=start)
    scene=movie.scenes[idx]
    credits,photos=cities
    im=Image.new('RGBA',(W,H),(9,27,46,255))
    im.alpha_composite(photos[idx],(0,124))
    base=ImageDraw.Draw(im)
    base.rectangle((0,0,W,123),fill=(8,24,40,255))
    # Photo credit stays on screen with the photo; it is explicitly archival.
    base.rectangle((0,587,W,664),fill=(4,20,35,225))
    entry=credits[idx]
    text(base,(24,590),f"FOTO: {entry['creator']}  •  {entry['license']}  •  RECORTE",F[18],(241,247,252,255))
    text(base,(24,613),entry['source_short'].replace('https://',''),F[18],(215,229,240,255))
    text(base,(24,637),entry['license_url'].replace('https://',''),F[18],(215,229,240,255))

    # Large, human presenter: photo may be replaced by any suitable portrait.
    rgba=cv2.cvtColor(portrait,cv2.COLOR_BGR2RGBA)
    rgba[:,:,3]=alpha
    character=Image.fromarray(rgba,'RGBA')
    width=540; height=round(width*character.height/character.width)
    character=character.resize((width,height),Image.Resampling.LANCZOS)
    im.alpha_composite(character,(210,662))

    overlay=Image.new('RGBA',(W,H)); d=ImageDraw.Draw(overlay)
    draw_logo(d,24,30)
    text(d,(695,43),date_label(movie.date),F[18],(224,237,246,255),anchor='ra')
    d.rounded_rectangle((22,141,225,176),radius=5,fill=(8,24,40,210))
    text(d,(32,147),'IMAGEM DE ARQUIVO',F[18],(241,247,252,255))
    # The forecast and presenter share a lower news-style panel while the
    # actual photograph remains clear and recognizable above them.
    d.rectangle((0,664,6,1135),fill=(255,202,65,255))
    if scene['name']=='PONTA DELGADA':
        text(d,(31,690),'PONTA',B[38]); text(d,(31,737),'DELGADA',B[38])
    else:
        text(d,(31,720),scene['name'],B[38])
    text(d,(28,795),f"{scene['temp']}°",B[126])
    text(d,(33,946),'MÁXIMA PREVISTA',F[18],(197,218,232,255))
    draw_weather_icon(d,weather_kind(scene['code']),69,1015,t)
    label=short_label(scene['code'])
    import textwrap
    for i,line in enumerate(textwrap.wrap(label,width=15,break_long_words=False)):
        text(d,(31,1065+i*25),line,B[20],(238,246,251,255))
    d.rectangle((0,1175,W,H),fill=(6,20,34,238))
    text(d,(24,1184),'DADOS: IPMA  •  APRESENTAÇÃO E VOZ SINTÉTICAS',F[18],(242,247,251,255))
    if movie.updated:
        stamp=movie.updated.replace('T',' ')[:16]
        text(d,(24,1209),f'Atualização IPMA: {stamp}',F[18],(215,229,240,255))
    if sample_credit:
        text(d,(24,1234),'RETRATO: LAMBDAWALKER · CC BY 4.0',F[18],(215,229,240,255))
        text(d,(24,1257),'creativecommons.org/licenses/by/4.0',F[18],(215,229,240,255))
    im.alpha_composite(overlay)
    return im.convert('RGB')


def render_video(movie, photo_path, model_path, output, work, lip_fps=12, batch_size=8):
    # Import MLX only here: preview and data validation do not need the GPU.
    import mlx.core as mx
    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline
    if FPS % lip_fps:
        raise ValueError(f'Lip FPS must divide {FPS}')
    mx.set_default_device(mx.gpu)
    source,alpha,(x1,y1,x2,y2),mask=load_photo(photo_path)
    cities=load_city_assets()
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
                    if t<movie.total: proc.stdin.write(composite_frame(movie,t,frame,alpha,cities,sample_credit).tobytes())
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
    p.add_argument('--lip-fps',type=int,default=24,choices=[12,24])
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
            source,alpha,_,_=load_photo(args.portrait)
            sample_credit=args.portrait.resolve()==(BASE/'assets/presenter-sample.png').resolve()
            composite_frame(movie,args.preview,source,alpha,load_city_assets(),sample_credit).save(args.output)
            print(f'Wrote preview {args.output}',flush=True)
        else:
            started=time.monotonic()
            render_video(movie,args.portrait,args.model.resolve(),args.output.resolve(),work,lip_fps=args.lip_fps)
            print(f'Total render time {time.monotonic()-started:.1f}s (excludes speech generation)',flush=True)


if __name__=='__main__':
    main()
