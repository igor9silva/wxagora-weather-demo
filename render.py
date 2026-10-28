#!/usr/bin/env python3
"""Render a vertical, data-driven weather short from local art and Kokoro clips.

Needs Pillow, NumPy, soundfile and ffmpeg. Source data and speech are separate
from the motion design. This is a demo, not an automated weather service.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from make_speech import line_for

BASE = Path(__file__).resolve().parent
W, H, FPS = 720, 1280, 24
RATE = 24000
INTRO, OUTRO = 0.8, 1.2
PLACES = [(1131200, "PORTO"), (1110600, "LISBOA"), (1080500, "FARO"), (3420300, "PONTA DELGADA")]
MONTHS = ["", "JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
WEEKDAYS = ["SEGUNDA", "TERÇA", "QUARTA", "QUINTA", "SEXTA", "SÁBADO", "DOMINGO"]
WHITE = (255, 255, 255, 255)
INK = (13, 34, 66, 255)
GOLD = (255, 202, 65, 255)


def font(size, heavy=False):
    paths = (["/System/Library/Fonts/Supplemental/Arial Black.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
             if heavy else ["/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"])
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if heavy else "DejaVuSans.ttf", size)


F = {s: font(s) for s in (18, 20, 22, 24, 26, 28, 30, 32, 38)}
B = {s: font(s, True) for s in (18, 20, 22, 24, 26, 28, 32, 38, 50, 60, 72, 90, 126)}


def clamp(x, a=0, b=1):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x*x*(3-2*x)


def fade_scene(u, duration):
    return min(ease(u/0.35), ease((duration-u)/0.22))


def short_label(code):
    return {
        0: "SEM INFORMAÇÃO", 1: "CÉU LIMPO", 2: "POUCO NUBLADO", 3: "PARCIALMENTE NUBLADO",
        4: "MUITO NUBLADO", 5: "NUVENS ALTAS", 6: "AGUACEIROS OU CHUVA", 7: "CHUVA FRACA",
        8: "AGUACEIROS FORTES", 9: "CHUVA OU AGUACEIROS", 10: "CHUVISCO", 11: "CHUVA FORTE",
        12: "PERÍODOS DE CHUVA", 13: "CHUVA FRACA", 14: "CHUVA FORTE", 15: "CHUVISCO",
        16: "NEBLINA", 17: "NEVOEIRO", 18: "NEVE", 19: "TROVOADA", 20: "AGUACEIROS E TROVOADA",
        21: "GRANIZO", 22: "GEADA", 23: "CHUVA E TROVOADA", 24: "NUVENS CONVECTIVAS",
        25: "PERÍODOS NUBLADOS", 26: "NEVOEIRO", 27: "NUBLADO", 28: "AGUACEIROS DE NEVE",
        29: "CHUVA E NEVE", 30: "CHUVA E NEVE",
    }.get(code, "VER PREVISÃO IPMA")


def weather_kind(code):
    if code in (6,7,8,9,10,11,12,13,14,15,19,20,21,23,28,29,30):
        return "rain"
    if code in (2,3):
        return "partly"
    if code in (4,5,16,17,24,25,26,27):
        return "cloud"
    return "sun"


def text(draw, xy, value, f, fill=WHITE, anchor=None, stroke=0, stroke_fill=(3, 16, 33, 170)):
    draw.text(xy, value, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_fill)


def fit_font(value, max_width, candidates):
    for s in candidates:
        if B[s].getlength(value) <= max_width:
            return B[s]
    return B[candidates[-1]]


def draw_logo(d, x=42, y=42):
    d.rounded_rectangle((x,y,x+51,y+51), radius=15, fill=(255,255,255,240))
    d.ellipse((x+10,y+18,x+27,y+35), fill=(255,183,27,255))
    d.ellipse((x+20,y+20,x+38,y+37), fill=(61,136,219,255))
    d.rounded_rectangle((x+11,y+29,x+41,y+39), radius=5, fill=(61,136,219,255))
    text(d, (x+64,y+2), "WXAGORA", B[24])
    text(d, (x+65,y+30), "CONCEITO INDEPENDENTE", F[18], (222,238,255,240))


def draw_weather_icon(d, kind, cx, cy, t):
    if kind in ("sun", "partly"):
        sx = cx if kind == "sun" else cx+12
        sy = cy if kind == "sun" else cy-9
        for i in range(8):
            a = 2*math.pi*i/8 + t*0.1
            d.line((sx+25*math.cos(a),sy+25*math.sin(a),sx+33*math.cos(a),sy+33*math.sin(a)),fill=(255,213,86,255),width=4)
        d.ellipse((sx-21,sy-21,sx+21,sy+21),fill=(255,205,65,255))
    if kind in ("cloud", "rain", "partly"):
        d.ellipse((cx-33,cy-3,cx+4,cy+26),fill=(240,247,255,255))
        d.ellipse((cx-20,cy-24,cx+18,cy+25),fill=(252,254,255,255))
        d.ellipse((cx+1,cy-13,cx+34,cy+25),fill=(232,241,253,255))
        d.rounded_rectangle((cx-25,cy+6,cx+27,cy+25),radius=10,fill=(241,248,255,255))
        if kind == "rain":
            for dx in (-19,0,19):
                d.line((cx+dx,cy+31,cx+dx-5,cy+40),fill=(101,205,255,255),width=4)


def date_label(date):
    from datetime import date as date_cls
    d = date_cls.fromisoformat(date)
    return f"{WEEKDAYS[d.weekday()]}  •  {d.day:02d} {MONTHS[d.month]} {d.year}"


class Movie:
    def __init__(self, forecast_path, audio_dir=None):
        self.path = Path(forecast_path)
        raw = json.loads(self.path.read_text())
        byid = {int(x["globalIdLocal"]): x for x in raw["data"]}
        self.date = raw["forecastDate"]
        self.updated = raw.get("dataUpdate", "")
        self.scenes = []
        self.audio_dir = Path(audio_dir) if audio_dir else BASE / "audio" / self.date
        manifest = json.loads((self.audio_dir / "manifest.json").read_text())
        for i,(ident,name) in enumerate(PLACES):
            v = byid[ident]
            clip = manifest[i]
            assert clip["place"].upper() == name, f"Mismatched narration: {name}"
            expected = line_for(i, name, round(float(v["tMax"])), int(v["idWeatherType"]))
            if clip["text"] != expected:
                raise ValueError(f"Stale narration for {name}; run make_speech.py again")
            seconds = max(2.70, float(clip["duration"]) + 0.32)
            self.scenes.append({"name":name,"temp":round(float(v["tMax"])),"code":int(v["idWeatherType"]),"duration":seconds,"clip":clip})
        self.starts = []
        now = INTRO
        for s in self.scenes:
            self.starts.append(now)
            now += s["duration"]
        self.scene_end = now
        self.total = now + OUTRO
        self.bg = Image.open(BASE/"assets/portugal-world.png").convert("RGB")
        self.mascot = Image.open(BASE/"assets/cloud-explorer.png").convert("RGBA")
        self.gradient = self.make_gradient()
        self.mascot_cache = {}

    def make_gradient(self):
        # A subtle top/bottom scrim keeps typography readable over every crop.
        a = np.zeros((H,W,4),dtype=np.uint8)
        a[:,:,:3] = [5,22,45]
        for y in range(H):
            a[y,:,3] = int(max(0, 150*(1-y/480))) if y < 480 else int(max(0, 150*((y-900)/380)))
        return Image.fromarray(a,"RGBA")

    def get_mascot(self, width, angle):
        w = int(round(width/4)*4)
        a = int(round(angle/2)*2)
        key = (w,a)
        if key not in self.mascot_cache:
            h = int(w*self.mascot.height/self.mascot.width)
            img = self.mascot.resize((w,h), Image.Resampling.LANCZOS)
            self.mascot_cache[key] = img.rotate(a, resample=Image.Resampling.BICUBIC, expand=True)
        return self.mascot_cache[key]

    def background(self, idx, u, transition=1):
        # Crops travel north to south through a single stylized illustration.
        srcw, srch = self.bg.size
        settings = [(0.41,0.27),(0.64,0.43),(0.72,0.67),(0.30,0.78)]
        if idx < 0:
            settings = [(0.50,0.48)]
            idx = 0
        cx, cy = settings[idx]
        if idx > 0 and transition < 1:
            px, py = settings[idx-1]
            cx = px + (cx-px)*transition
            cy = py + (cy-py)*transition
        zoom = 1.62 + 0.045*clamp(u/3.4)
        crop_h = srch/zoom
        crop_w = crop_h*W/H
        x = clamp(cx*srcw + 9*math.sin(u*.35)-crop_w/2, 0, srcw-crop_w)
        y = clamp(cy*srch + 10*math.sin(u*.25)-crop_h/2, 0, srch-crop_h)
        image = self.bg.crop((int(x),int(y),int(x+crop_w),int(y+crop_h))).resize((W,H),Image.Resampling.BILINEAR).convert("RGBA")
        image.alpha_composite(self.gradient)
        return image

    def mascot_at(self, im, x, y, width, angle=0, alpha=255):
        mascot = self.get_mascot(width, angle)
        if alpha < 255:
            mascot = mascot.copy()
            channel = mascot.getchannel("A").point(lambda p:int(p*alpha/255))
            mascot.putalpha(channel)
        im.alpha_composite(mascot, (int(x-mascot.width/2), int(y-mascot.height/2)))

    def atmosphere(self, im, code, t, idx):
        overlay = Image.new("RGBA",(W,H)); d=ImageDraw.Draw(overlay)
        kind = weather_kind(code)
        if kind == "rain":
            d.rectangle((0,0,W,H),fill=(20,55,100,35))
            # Deterministic illustrative rain, never represented as radar.
            for j in range(60):
                x=(j*113+43*idx)%W
                y=int((j*191+t*390)%H)
                d.line((x,y,x-8,y+22),fill=(196,226,255,75),width=2)
        if kind in ("cloud", "partly"):
            d.ellipse((350+12*math.sin(t*.3),135,590+12*math.sin(t*.3),219),fill=(197,218,235,62))
            d.ellipse((429+12*math.sin(t*.3),96,603+12*math.sin(t*.3),210),fill=(216,230,241,65))
        if kind in ("sun", "partly"):
            pulse = 0.5+0.5*math.sin(t*2)
            d.ellipse((558,113,638,193),fill=(255,226,129,int(35+12*pulse)))
        # Tiny flecks hint at travel between the towns.
        for j in range(15):
            x=(j*147 + 39)%W; y=int((j*107-t*38)%H)
            r=2+(j%3==0)
            d.ellipse((x-r,y-r,x+r,y+r),fill=(235,248,255,100))
        im.alpha_composite(overlay)

    def frame(self, t):
        if t < INTRO:
            im = self.background(-1,t)
            overlay = Image.new("RGBA",(W,H)); d = ImageDraw.Draw(overlay)
            draw_logo(d)
            text(d,(43,210),"PORTUGAL",B[90],stroke=2)
            text(d,(46,316),"EM VIAGEM",B[60],(255,213,97,255),stroke=2)
            text(d,(47,405),date_label(self.date),B[22],(245,250,255,255),stroke=1)
            im.alpha_composite(overlay)
            p=ease(t/.8)
            self.mascot_at(im, 570-125*p, 850-35*p, 355, -8+7*p)
            return im.convert("RGB")
        if t >= self.scene_end:
            im = self.background(-1,t)
            overlay=Image.new("RGBA",(W,H)); d=ImageDraw.Draw(overlay)
            d.rectangle((0,0,W,H), fill=(7,26,50,160))
            draw_logo(d)
            text(d,(46,190),"PORTUGAL",B[72],stroke=1)
            text(d,(48,275),date_label(self.date),B[22],(221,238,255,255))
            for i,s in enumerate(self.scenes):
                y=374+i*142
                d.rounded_rectangle((42,y,678,y+108),radius=25,fill=(255,255,255,235))
                text(d,(68,y+18),s["name"],fit_font(s["name"],360,[38,32,28,24]),INK)
                text(d,(68,y+66),short_label(s["code"]),F[18],(46,83,116,255))
                text(d,(646,y+52),f'{s["temp"]}°',B[50],INK,anchor="rm")
            text(d,(48,1130),"MÁXIMAS PREVISTAS  •  FONTE: IPMA",B[20])
            text(d,(48,1170),"CONCEITO VISUAL  •  DADOS DE DEMONSTRAÇÃO",F[18],(225,238,250,255))
            im.alpha_composite(overlay)
            return im.convert("RGB")
        idx=max(i for i,start in enumerate(self.starts) if t>=start)
        s=self.scenes[idx]; u=t-self.starts[idx]; dur=s["duration"]
        travel=ease(u/.55)
        im=self.background(idx,u,travel)
        self.atmosphere(im,s["code"],t,idx)
        overlay=Image.new("RGBA",(W,H)); d=ImageDraw.Draw(overlay)
        draw_logo(d)
        text(d,(44,132),date_label(self.date),B[20],(236,246,255,255),stroke=1)
        # The title and numbers are kept in graphics, not AI-generated imagery.
        a=int(255*fade_scene(u,dur)); ac=(255,255,255,a)
        title_font=fit_font(s["name"],620,[72,60,50,38])
        dy=int(17*(1-ease(u/.4)))
        text(d,(43,199+dy),s["name"],title_font,ac,stroke=2)
        text(d,(47,296+dy),f'{s["temp"]}°',B[126],ac,stroke=2)
        text(d,(54,447+dy),"MÁXIMA PREVISTA",B[20],(244,248,255,a),stroke=1)
        badge_y=501+dy
        d.rounded_rectangle((44,badge_y,587,badge_y+84),radius=25,fill=(7,34,64,int(a*.78)))
        draw_weather_icon(d,weather_kind(s["code"]),89,badge_y+41,t)
        label=short_label(s["code"])
        text(d,(144,badge_y+26),label,fit_font(label,415,[26,24,22,20,18]),(255,255,255,a))
        # Animated teleport halo and travel indicator.
        bounce=math.sin(t*3.2)
        arriving=ease(u/.55)
        departing=ease((dur-u)/.5)
        px=500+24*math.sin(t*1.4+idx)-220*(1-arriving)+190*(1-departing)
        py=829+13*bounce-120*(1-arriving)-175*(1-departing)
        d.ellipse((px-142,py+156,px+142,py+190),outline=(103,223,255,200),width=4)
        d.ellipse((px-110,py+162,px+110,py+185),outline=(232,251,255,145),width=2)
        phase=min(ease(u/.4),ease((dur-u)/.38))
        if phase<.95:
            rr=int(85+75*(1-phase))
            d.ellipse((px-rr,py-rr,px+rr,py+rr),outline=(148,234,255,int(180*(1-phase))),width=7)
            for k in range(7):
                xx=80+k*100
                yy=680+k*31
                d.line((xx-40,yy+35,xx+45,yy-30),fill=(189,240,255,int(125*(1-phase))),width=3)
        # Progress dots and source in the safe lower region.
        for j in range(4):
            x=48+j*25
            d.ellipse((x,1122,x+9,1131),fill=(255,211,92,255) if j==idx else (255,255,255,120))
        text(d,(44,1160),"FONTE: IPMA  •  DEMONSTRAÇÃO",B[18],(247,250,255,245),stroke=1)
        if self.updated:
            stamp=self.updated.replace("T"," ")[:16]
            text(d,(44,1192),f"Atualização da fonte: {stamp}",F[18],(235,244,255,230),stroke=1)
        im.alpha_composite(overlay)
        self.mascot_at(im,px,py,230+100*phase,4*math.sin(t*2)+15*(1-arriving)-15*(1-departing),int(255*clamp(phase+.1)))
        return im.convert("RGB")

    def write_audio(self, output, include_music=True):
        n=int(math.ceil(self.total*RATE)); audio=np.zeros(n,dtype=np.float64)
        # Original, low-volume instrumental bed: soft chords and mallet notes.
        notes=[261.63,329.63,392.00,523.25,392.00,329.63,293.66,440.00]
        if include_music:
            for beat in range(int(self.total/.43)+1):
                start=int(beat*.43*RATE); length=min(int(.7*RATE),n-start)
                if length<=0: continue
                t=np.arange(length)/RATE
                env=np.exp(-t*7)*(1-np.exp(-t*100))
                frequency=notes[beat%len(notes)]
                audio[start:start+length] += .026*env*(np.sin(2*np.pi*frequency*t)+.25*np.sin(2*np.pi*frequency*2*t))
                if beat%2==0:
                    pulse=np.exp(-t*18)*np.sin(2*np.pi*(72*t-24*t*t))
                    audio[start:start+length] += .038*pulse
        for idx,s in enumerate(self.scenes):
            wav,sr=sf.read(self.audio_dir/s["clip"]["file"],dtype="float64")
            if sr!=RATE: raise ValueError(f"Unexpected sample rate {sr}")
            if wav.ndim>1: wav=wav.mean(axis=1)
            peak=max(np.max(np.abs(wav)),1e-6)
            wav=wav*(.72/peak)
            start=int((self.starts[idx]+.18)*RATE)
            ncopy=min(len(wav),n-start)
            audio[start:start+ncopy]+=wav[:ncopy]
            # Subtle synthesized transition chime.
            if include_music:
                length=int(.35*RATE); tt=np.arange(length)/RATE
                swoosh=np.exp(-tt*9)*np.sin(2*np.pi*(410*tt+480*tt*tt))*.025
                p=max(0,start-int(.16*RATE)); q=min(n-p,length)
                audio[p:p+q]+=swoosh[:q]
        audio=np.clip(audio,-.97,.97)
        sf.write(output,audio,RATE,subtype="PCM_16")

    def render(self, output):
        output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        wav=output.with_suffix(".mix.wav")
        self.write_audio(wav)
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{W}x{H}","-r",str(FPS),"-i","-","-i",str(wav),"-c:v","libx264","-preset","veryfast",
             "-crf","19","-pix_fmt","yuv420p","-c:a","aac","-b:a","160k","-movflags","+faststart",
             "-shortest",str(output)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        total_frames=math.ceil(self.total*FPS)
        try:
            for i in range(total_frames):
                proc.stdin.write(self.frame(i/FPS).tobytes())
                if i%120==0: print(f"{self.date}: {i}/{total_frames} frames",flush=True)
        finally:
            proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError("ffmpeg failed")
        wav.unlink(missing_ok=True)
        print(f"Wrote {output} ({self.total:.1f}s)",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("forecast",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--preview",type=float,help="Write one frame as a PNG at this timestamp")
    args=p.parse_args()
    movie=Movie(args.forecast)
    if args.preview is not None:
        movie.frame(args.preview).save(args.output)
        print("Wrote preview",args.output,"duration",movie.total)
    else:
        movie.render(args.output)


if __name__ == "__main__":
    main()
