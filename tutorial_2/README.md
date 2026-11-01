# 2단계 최종 PPO · 팀 1~5단계 공통 규격

이 폴더 전체를 **`Pygame-Pingu-Ball-/tutorial_2/`**에 두세요. **명령은 반드시 프로젝트 최상위**(`Pygame-Pingu-Ball-`)에서 실행합니다. `rl/`, `pingu_ball/`, `tools/`, `experiments/tutorial_1_5/`, `tutorial_2_4/`는 그 최상위에 존재해야 합니다.

> 기존 팀의 1·3·5·4단계 PPO 체크포인트 및 코드 변경 없음. 2단계만 개인 보상을 사용한 PPO로 교체한 **단계별 전문가 선택 실행기**입니다. 단일 신경망으로 5개 맵을 학습한 것은 아닙니다.

## 검증된 최종 모델

- PPO, 3행동(좌·대기·우), 6프레임, 팀의 `MilestoneEnv` 공통 관측
- 12프레임 **순수 PPO**로 처음부터 학습한 정책을 6프레임 순수 PPO로 추가 학습. BC/DAgger/사람 시범 미사용
- 공통 성공 +100 / 실패 −25 유지, 2단계 점프 목표 착지·높이·수평 보상을 **팀의 구간 보상 대신** 지급 (중복 지급 없음)
- `results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42/best.zip`: 최종 모델 (이 README 기준 `tutorial_2/` 안의 상대 경로)
- 종전 평가: 200판 중 192회 클리어 = 96% (seed 120000); 통합 실행 100판 95%; 실제 게임 화면 1회 클리어 확인. **이 결과는 이동 전 PC에서 측정했으며, Git 새 경로에서의 재검증은 아래 명령으로 확인해야 합니다.**

## 실행

```cmd
python -m tutorial_2.stage2_common_bridge.run verify --steps 100
python -m tutorial_2.stage2_common_bridge.run evaluate --stage 2 --episodes 200 --seed 120000
python -m tutorial_2.stage2_common_bridge.run play --stage 2 --seed 120000
python -m tutorial_2.stage2_common_bridge.run evaluate --stage all --episodes 100 --seed 120000
```

2단계만 학습 당시 코드로 평가하는 명령:

```cmd
python -m tutorial_2.reward_transfer_2_6.train evaluate --trials 200 --eval-seed 120000
```

필요한 패키지는 프로젝트의 `requirements-rl.txt`를 참조하세요. `python -m pytest -q tutorial_2/reward_repeat_2_4/tests tutorial_2/reward_transfer_2_6/tests tutorial_2/stage2_common_bridge/tests`로 정적/회귀 테스트를 실행할 수 있습니다.

## 구조

| 경로 | 내용 |
|---|---|
| `reward_repeat_2_4/` | 12프레임 원본 PPO 학습 및 2단계 개인 보상(실행에 필요) |
| `reward_transfer_2_6/` | 12→6프레임 PPO 가중치 이어 학습, 최종 모델 평가 |
| `stage2_common_bridge/` | 1~5단계 전문가 선택 실행기·검증·화면 재생 |
| `tutorial_test/` | 과거 팀 보상 비교 실험 (최종 96% 모델에는 직접 필요 없음) |
| `results/` | 저장된 학습 체크포인트·평가 기록 |

이동 후 기본 경로는 전부 `tutorial_2/results/` 아래로 설정했습니다. JSON 설정과 평가 기록의 과거 Windows 절대 경로는 **이동 후 `tutorial_2/results/` 상대 경로로 표기만 정리**했습니다. 점수·성공률과 PPO 체크포인트 바이너리는 바꾸지 않았습니다. 실행 시 체크포인트 경로는 코드에서 폴더 위치 기준으로 구합니다. PPO `best.zip`은 수정하지 않았습니다. 1·3·5·4단계 원본은 팀 `tools/tutorial_rl.py`를 통해 원래 고정 런타임으로 실행합니다.

**Git 업로드 주의:** `results/`에는 실패했던 과거 모델과 중복 저장본도 들어있습니다. 최종 실행에 필요한 핵심 파일은 `results/reward_transfer_2_6/.../best.zip`, 12프레임 출발 모델 `results/reward_repeat_2_4/.../best.zip` 및 각 `config.json`입니다. 이전 결과는 보관용으로만 사용해도 됩니다.
