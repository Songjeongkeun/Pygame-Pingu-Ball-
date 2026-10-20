"""Measure simulation and actual Pygame rendering without the 60fps wait."""
from pathlib import Path
import argparse
import json
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.app import App
import pygame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages',type=int,nargs='+',default=[5,8,11,12],choices=range(1,13))
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--output',type=Path,default=ROOT/'output/performance/benchmark.json')
    args=parser.parse_args()
    if args.frames<20: parser.error('--frames must be at least 20')
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        app=App(Path(tmp)/'progress.json',headless=True)
        app.sound_on=False
        try:
            for number in args.stages:
                app.start(number)
                for _ in range(8): app.update(1/60,0);app.render()
                samples={'update':[],'render':[],'frame':[]}
                for _ in range(args.frames):
                    start=time.perf_counter();app.update(1/60,0)
                    middle=time.perf_counter();app.render();end=time.perf_counter()
                    samples['update'].append((middle-start)*1000)
                    samples['render'].append((end-middle)*1000)
                    samples['frame'].append((end-start)*1000)
                row=dict(stage=number,frames=args.frames)
                for key,values in samples.items():
                    row[key+'_mean_ms']=round(statistics.mean(values),3)
                    row[key+'_p95_ms']=round(sorted(values)[int((len(values)-1)*.95)],3)
                results.append(row);print(json.dumps(row),flush=True)
        finally: pygame.quit()
    report=dict(headless=True,window=[1000,520],render_hz=60,physics_hz=120,
                warmup_frames=8,direction=0,sound=False,results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
