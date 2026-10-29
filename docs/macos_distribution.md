# macOS 배포

`.github/workflows/build-macos.yml`은 Apple Silicon(arm64, macOS 14 러너)과
Intel(x86_64, macOS 15 러너)용 `PinguBall.app`을 각각 만든다.
Windows에서 파일 확장자를 바꾸거나 교차 빌드하지 않고 실제 macOS 환경의
Python 3.12와 현재 프로젝트 의존성으로 빌드한다.

`main` 또는 `design-수정` 브랜치의 게임·리소스·빌드 파일을 푸시하면 실행한다.
기본 브랜치에 워크플로가 반영된 뒤에는 GitHub Actions의 **Build macOS →
Run workflow**에서 대상 브랜치를 선택해 수동으로 다시 실행할 수도 있다.

성공한 실행의 Artifacts에서 다음 파일을 받는다.

| 대상 Mac | Artifact | 내부 배포 ZIP |
|---|---|---|
| M1·M2·M3 등 Apple Silicon | PinguBall-macOS-AppleSilicon | PinguBall-macOS-AppleSilicon.zip |
| Intel Mac | PinguBall-macOS-Intel | PinguBall-macOS-Intel.zip |

Artifact를 내려받아 압축을 풀고 내부 ZIP도 풀면 `PinguBall.app`과
`사용방법.txt`가 나온다. Python 설치 없이 `.app` 전체를 이동해 실행한다.
ZIP은 macOS `ditto`로 만들어 실행 권한과 번들 내부 심볼릭 링크를 보존한다.
빌드 메타데이터 JSON에는 아키텍처·Python·macOS 버전과 ZIP의 SHA-256을 남긴다.

이 빌드는 PyInstaller의 ad-hoc 서명을 사용하며 Apple Developer ID 서명과
Apple 공증은 하지 않는다. 처음 실행을 macOS가 차단하는 경우 해당 앱에 대해
시스템 설정의 개인정보 보호 및 보안에서 허용 여부를 선택한다. 시스템 전체
보안 설정을 끄는 절차는 필요하지 않다.

한글은 재배포 가능한 나눔고딕 Regular·Bold와 OFL 라이선스를 앱에 함께 넣는다.
Windows는 기존 맑은 고딕을 우선 사용하므로 기존 Windows 화면은 유지된다.
macOS 실행 파일의 기록은 앱 번들 밖의 사용자 폴더에 저장한다.

```text
~/Library/Application Support/PinguBall/progress.json
```

각 Mac 러너에서 전체 단위 테스트를 실행하고, 빌드한 앱을 작업 폴더와 다른
위치에서 실행해 타이틀·일반 7번·튜토리얼 10번 화면을 렌더링한다. 실행 파일의
아키텍처와 앱 서명도 검사한다. 키보드·오디오 장치를 사용하는 실제 Mac에서의
수동 플레이는 별도 확인이 필요하며, 더 오래된 macOS 버전의 호환성은 보장하지 않는다.

직접 Mac에서 빌드하려면 프로젝트 루트에서 실행한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
python tools/build_macos.py --arch arm64
python tools/verify_macos.py --arch arm64 --app output/macos/PinguBall-macOS-AppleSilicon/PinguBall.app
```

Intel Mac에서는 두 명령의 `arm64`를 `x86_64`로, 경로의 `AppleSilicon`을 `Intel`로 바꾼다.
