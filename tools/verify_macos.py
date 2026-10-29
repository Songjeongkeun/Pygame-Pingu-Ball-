"""Validate the native application, its resources, signature and rendering."""
import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=Path,required=True)
    parser.add_argument('--arch',choices=('arm64','x86_64'),required=True)
    args=parser.parse_args()
    if sys.platform!='darwin':parser.error('Verification requires macOS.')
    app=args.app.resolve()
    executable=app/'Contents'/'MacOS'/'PinguBall'
    with (app/'Contents'/'Info.plist').open('rb') as file:
        info=plistlib.load(file)
    assert info['CFBundleExecutable']=='PinguBall'
    assert os.access(executable,os.X_OK)
    arches=subprocess.check_output(['lipo','-archs',str(executable)],text=True).split()
    assert args.arch in arches,(args.arch,arches)
    root=Path(__file__).resolve().parents[1]
    output=root/'output'/'macos'/f'verification-{args.arch}'
    output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
    for name,options in (('title',[]),('stage7',['--stage','7']),('tutorial10',['--special','10'])):
        shot=output/f'{name}.png'
        subprocess.run([str(executable),'--headless','--frames','3','--screenshot',str(shot),*options],
                       cwd=output,env=env,check=True,timeout=60)
        assert shot.is_file() and shot.stat().st_size>10000,name
    # Launching from an unrelated working directory must not alter the bundle.
    subprocess.run(['codesign','--verify','--deep','--strict',str(app)],check=True)
    print(f'Native {args.arch} app: title, stage 7, tutorial 10 and signature verified.')


if __name__=='__main__':main()
