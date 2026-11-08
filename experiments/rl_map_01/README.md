# 복합 맵 1: 계단에서 2층으로 넘어가기

하층에서는 오른쪽으로 가다가 마지막 계단에서 왼쪽으로 돌아 2층에 올라가야 한다.
원래 출발점에서만 학습한 정책은 독립 시험 100판 중 99판에서 층 전환 구간에 이르렀지만, 2층에 진입하지 못했다.
[실패 진단](diagnostics/floor-transition.ko.md)은 방향 선택과 행동 유지 시간을 추적하고, 같은 물리 상태에서 기존 행동으로 연결 착지가 가능한지 검사한다.
이 수동 진단은 학습에 사용하지 않았다.

| 항목 | 원래 출발점에서 학습 | 커리큘럼으로 학습 |
| --- | ---: | ---: |
| 학습 시드 | 2026 | 2026 |
| 실제 행동 스텝 | 500,224 | 500,224 |
| 선택 모델 저장 시점 | 500,224 | 350,208 |
| 선택 모델 완주 | 0/100 | 49/100 |
| 선택 모델 2층 진입 | 0/100 | 93/100 |
| 선택 모델 평균 구간 달성 | 17.88/33 | 28.71/33 |
| 마지막 모델 완주 | 0/100 | 0/100 |

두 실행 모두 PPO, 9개 행동, 1,984차원 관측, 구간별 최초 +1(최대 33개), 이글루 +100, 실패·시간 초과 −25를 사용했다.
커리큘럼 실행은 정책이 학습 중 실제로 도달한 계단과 2층 상태를 일부 환경의 출발점으로 다시 사용했다.
검증과 독립 시험은 두 실행 모두 원래 출발점에서 진행했다. 모델은 검증으로 선택했고 독립 시험은 확률 표본 방식이었다.
결과는 학습 시드 하나에 대한 비교다.

## 보존 파일

| 위치 | 자료 |
| --- | --- |
| [original_start](original_start/report.ko.md), [curriculum_start](curriculum_start/report.ko.md) | 각 실행의 원본 보고서, `plan.json`, `versions.json`, 설정과 선택 검증, 진행표, 독립 시험과 한 판 재생 결과가 든 `summary.json` |
| 각 실행의 `seed_2026/best.zip`, `seed_2026/latest.zip` | 검증 선택 모델과 마지막 모델. 파일명을 유지해 기존 학습 도구의 구조와 맞췄다. |
| 각 실행의 `source_snapshot/` | 당시 학습 소스와 맵 배치. 최신 작업 코드와 달라질 수 있어 두 실행의 파일을 각각 보존했다. |
| [diagnostics](diagnostics/floor-transition.ko.md) | 층 전환 실패 진단과 JSON 근거 |
| [figures](figures/) | 두 실행의 학습 지표·구간 보상·PPO 지표 그래프, 커리큘럼 시작 유형과 계단 방향 선택 그래프 |
| [sha256.json](sha256.json) | 복사한 증거 파일의 SHA-256. 완료 기록의 요약 경로만 상대 경로로 바꿨다. |

전체 에피소드 로그, 초기 모델, 중간 상태의 직렬화 파일, 영상 원본은 이 묶음에 넣지 않았다.
커리큘럼 상태의 개수와 구간별 통계는 `states_manifest.json`과 `summary.json`에서 확인할 수 있다.
기존 로컬 `output/` 원본은 그대로 보관한다.

## 한 판 재생하기

저장소 루트에서 [실험 목차](../../docs/rl-experiments.md)의 Python 환경을 준비한 뒤 실행한다.

```powershell
.venv/Scripts/python.exe -B experiments/rl_map_01/replay.py
.venv/Scripts/python.exe -B experiments/rl_map_01/replay.py original_start
.venv/Scripts/python.exe -B experiments/rl_map_01/replay.py curriculum_start
.venv/Scripts/python.exe -B experiments/rl_map_01/replay.py curriculum_start --checkpoint latest
```

첫 명령은 파일 해시만 검사한다. 나머지는 각 보존본의 소스에서 모델을 불러 최대 확률 행동으로 한 판을 재생한다.
화면이나 FFmpeg 없이 원래 게임 물리와 비교하고, 완주 여부·보상·시간을 저장된 기록과 대조한다.
커리큘럼 선택본 시연은 **36.37초, 33/33개, 133점**으로 끝난 한 판이다. 위의 100판 시험과는 별개다.
