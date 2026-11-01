# 튜토리얼 1~5 강화학습

현재 코드의 보상 환경·A2C/PPO 학습·평가·재생 모듈이다.
기존 실험은 [실험 묶음](../experiments/tutorial_1_5/README.md)에 정리했다.
해당 문서에 성공률·그래프·변수·보상 규칙·실행 명령을 함께 설명한다.

## 코드 역할

| 파일 | 역할 |
|---|---|
| `milestone_env.py` | 성공 +100, 실패/180초 -25, 구간 최초 착지 +10 보상과 관측 |
| `milestone_models.py` | 핵심 구간 입력을 받는 객체 인코더와 A2C/PPO 정책 |
| `milestone_train.py` | 맵별 전문가 학습, 검증 선택, 교사 기준과 통합 절차 |
| `milestone_report.py`, `milestone_map.py` | 성공률·학습 곡선·보상 구간 보고서 |
| `milestone_watch.py`, `milestone_play.py` | 실제 상태를 관측해 선택 모델 재생 |
| `milestone_prevalidate.py`, `milestone_finish.py` | 교사 검증과 완료 후 검증·보고 |
| `recovery_train.py` | 2·4번 PPO 안정화·긴 수집·긴 행동 비교 |
| `recovery_report.py`, `recovery_watch.py` | 후속 실험 분석과 재생 |
| `five_env.py`, `five_models.py`, `five_train.py` | 공통 상태·신경망·경험 수집·균등 정책 통합 기반 |
| `five_report.py`, `five_watch.py` | 공통 보고·재생 |
| `all_tutorial_env.py`, `map_input_env.py` | 원래 게임 물리 환경과 맵 정보 입력 |

완료한 모델에는 학습 당시 소스의 해시가 기록되어 있다. 보존 모델은
`tools/tutorial_rl.py`가 각 실험의 고정 런타임에서 실행한다. 해시 검사를 우회하지 않는다.
현재 파일의 주석과 게임 맵은 과거 고정 소스와 다를 수 있으므로 원래 `best.zip`을
현재 `rl/` 코드로 직접 재생하는 대신 정리된 런처를 사용한다.

```powershell
.venv\Scripts\python.exe -B tools/tutorial_rl.py verify
.venv\Scripts\python.exe -B tools/tutorial_rl.py play --stage 5
```

기본 비교 실험은 PPO 1·3·5번 99%, A2C 전부 0%, PPO 2·4번 0%였다.
후속 2번 최선 조건은 시드 평균 54.3%이며 이 로컬 후속 실험의 4번은 0%였다.
통합 정책은 교사 품질 기준 미달로 실행하지 않았다.
원격에 따로 있는 `tutorial_2_4/`의 실험과는 결과를 구분한다.

원본 `output/rl_user_reward_20261030`과 `output/rl_recovery_20261030`은 로컬에 유지한다.
Git에는 그중 정리된 자료만 `experiments/tutorial_1_5`에 포함한다.
