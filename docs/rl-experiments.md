# 핑구볼 강화학습 실험

튜토리얼에서 A2C와 PPO를 비교한 뒤, 학습이 어려웠던 2·4번을 따로 실험했다.
복합 맵 1에서는 계단에서 2층으로 넘어가지 못하는 문제를 분석하고, 학습할 때의 시작 상태를 바꿨다.

튜토리얼을 실행하려면 **[튜토리얼 1~5 통합 실행 및 실험 보고](rl-tutorials.md)**부터 보면 된다.

| 실험 | 평가 조건 | 기록된 결과 |
| --- | --- | --- |
| [튜토리얼 1~5 통합](rl-tutorials.md) | 단계별 기존 PPO 모델을 공통 명령에서 평가·재생 | 새 평가 결과와 A2C/PPO 비교·2·4 후속 실험은 통합 보고서에서 구분한다. |
| [복합 맵 1: 층 전환과 커리큘럼](../experiments/rl_map_01/README.md) | 학습 시드 2026, 두 실행 각 500,224 행동 스텝. 선택 모델을 원래 출발점에서 확률 표본으로 독립 시험 100판 | 0/100 → 49/100. 2층 진입률 0% → 93%. 커리큘럼 마지막 모델은 0/100. |

튜토리얼 3번 기본 PPO는 성공한 99판 중 98판에서 지정 핵심 구간 보상을 얻지 않았다.
클리어와 의도한 경로 달성은 따로 본다. 학습 중 검증은 모델 선택용이며 독립 시험이나 시연 한 판과 합산하지 않는다.
평가 정책·시드·보상이 다른 실험의 숫자를 같은 조건의 성능 순위로 사용하지 않는다.

## 어디서 시작할까

저장소 루트에서 Python 3.12 환경을 사용한다. 아래 명령은 보존 자료를 검사하거나 재생할 뿐 학습하지 않는다.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-rl.txt
.venv/Scripts/python.exe -B tools/tutorial_rl.py verify --suite combined
.venv/Scripts/python.exe -B tools/tutorial_rl.py list --suite combined
.venv/Scripts/python.exe -B tools/tutorial_rl.py play --suite combined --stage 5
.venv/Scripts/python.exe -B experiments/rl_map_01/replay.py curriculum_start
```

macOS/Linux에서는 Python 경로를 `.venv/bin/python`으로 바꾼다.
튜토리얼 1·3·5번은 [기존 실험 보존본](../experiments/tutorial_1_5/README.md)의 고정 런타임과 모델을 사용한다.
맵 1 재생기는 해당 실행의 소스와 배치, 체크포인트를 해시로 확인하고 저장된 최대 확률 재생 한 판을 대조한다.
