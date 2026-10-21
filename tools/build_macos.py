"""Build and archive a native macOS application. Run on a Mac with Python 3.12."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
LABELS={'arm64':'AppleSilicon','x86_64':'Intel'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch',choices=LABELS,default=platform.machine())
    args=parser.parse_args()
    if sys.platform!='darwin':
        parser.error('Build on macOS, or run the Build macOS GitHub Actions workflow.')
    if platform.machine()!=args.arch:
        parser.error('Use native Python and a runner matching --arch.')
    label=LABELS[args.arch]
    output=ROOT/'output'/'macos'
    work=ROOT/'output'/'build'/f'macos-{args.arch}'
    package=output/f'PinguBall-macOS-{label}'
    package.mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    subprocess.run([
        sys.executable,'-m','PyInstaller','--noconfirm','--onedir','--windowed',
        '--name','PinguBall','--target-architecture',args.arch,
        '--osx-bundle-identifier','com.songjeongkeun.pinguball',
        '--codesign-identity','-',
        '--distpath',str(package),'--workpath',str(work/'work'),
        '--specpath',str(work),'--paths',str(ROOT),
        '--add-data',f'{ROOT / "assets"}{os.pathsep}assets',str(ROOT/'main.py'),
    ],cwd=ROOT,check=True)
    app=package/'PinguBall.app'
    # PyInstaller also emits an onedir staging folder. Zip only the .app and
    # instructions so its libraries aren't duplicated in the download.
    readme=package/'사용방법.txt'
    readme.write_text(
        f'핑구볼 macOS {label}\n\n'
        'PinguBall.app을 원하는 폴더로 옮겨 실행하세요. Python 설치는 필요 없습니다.\n'
        '이 버전은 Apple 개발자 서명·공증을 하지 않은 수업 배포용 빌드입니다.\n'
        '처음 실행할 때 macOS에서 차단하면 시스템 설정 > 개인정보 보호 및 보안에서\n'
        '이 앱에 한해 허용 여부를 선택할 수 있습니다.\n'
        '기록: ~/Library/Application Support/PinguBall/progress.json\n'
        '← → 이동 / R 재시작 / ESC 또는 P 일시정지\n'
        '대시 ←← 또는 →→ / 점프 ↑↑ / 마지막에 먹은 아이템 하나만 보유\n',encoding='utf-8')
    archive=output/f'PinguBall-macOS-{label}.zip'
    # ditto preserves the signed bundle's symlinks and executable permissions.
    subprocess.run(['ditto','-c','-k','--sequesterRsrc','--keepParent',str(app),str(archive)],check=True)
    subprocess.run(['zip','-j',str(archive),str(readme)],check=True)
    metadata={'architecture':args.arch,'python':platform.python_version(),
              'macos':platform.mac_ver()[0],'archive':archive.name,
              'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'signing':'ad-hoc; not notarized'}
    (output/f'build-{args.arch}.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(archive,flush=True)


if __name__=='__main__':main()
