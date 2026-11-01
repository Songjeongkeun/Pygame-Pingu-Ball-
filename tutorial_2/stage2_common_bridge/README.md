> **폴더 이동 안내:** 이 문서는 기존 실험 설명을 보존한 것이야. 실제 위치는 프로젝트 루트의 `tutorial_2/stage2_common_bridge/`이며 모든 명령은 프로젝트 루트에서 `python -m tutorial_2.stage2_common_bridge.run ...` 형식으로 실행해. 경로가 헷갈리면 `tutorial_2/README.md`를 우선 참고해.

# 팀 1~5 공통 실행기 + 개인 2단계 PPO (6프레임)

**용도:** 기존 팀 PPO 모델/보상을 덮어쓰지 않고 2단계만 개인 PPO로 연결해 평가·재생하기.
**기존 모델을 하나의 신경망으로 합치는 작업은 아닙니다.** 1·3·4·5는 팀의 고정 런타임에 보존된 모델을 선택하고, 2만 새 96% 모델을 선택합니다.

## 설치

최상위에 `tutorial_2/` 폴더를 배치하세요.
기존 `rl/`, `tools/`, `tutorial_2_4/`, `reward_repeat_2_4/`, `reward_transfer_2_6/`, `tutorial_2/results/`는 덮어쓰거나 삭제하지 마세요.

반드시 기존 6프레임 PPO 가중치가 아래에 있어야 합니다.

`tutorial_2/results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42/best.zip`

이미 사용 중인 `reward_repeat_2_4/`, `reward_transfer_2_6/`, `tutorial_2_4/` 폴더도 그대로 필요합니다. 팀의 보존 모델들을 실행하려면 `experiments/tutorial_1_5/`도 Git에 포함되어 있어야 합니다.

## 실행 (Windows cmd / venv)

프로젝트 최상위에서:

```cmd
python -m tutorial_2.stage2_common_bridge.run verify --steps 100
python -m tutorial_2.stage2_common_bridge.run evaluate --stage 2 --episodes 200 --seed 120000
python -m tutorial_2.stage2_common_bridge.run play --stage 2 --seed 120000
```

1·3·5 원본 모델을 유지한 채 묶음으로 추가 검증하려면:

```cmd
python -m tutorial_2.stage2_common_bridge.run verify --archive
python -m tutorial_2.stage2_common_bridge.run evaluate --stage all --episodes 100 --seed 120000
```

`--stage all`은 4단계도 **기존 팀 PPO 베이스라인**으로 실행합니다. 4단계 보상이나 모델은 바꾸지 않습니다. 별개 단계 결과를 하나의 JSON에 모읍니다. 기존 1·3·5 성공률 99%는 저장 당시 평가 기록이며, 새 시드의 결과는 달라질 수 있습니다.

다른 단계의 재생:

```cmd
python -m tutorial_2.stage2_common_bridge.run play --stage 3 --seed 120000
```

`--greedy`는 한 판의 최대 확률 행동 재생에만 사용하며 성공률 평가 프로토콜과 다릅니다.

## 실제 연결 방식

| 단계 | PPO 체크포인트 | 실행 관측·보상 |
|---|---|---|
| 1, 3, 5 | 팀 기존 보존 `baseline` PPO | `tools/tutorial_rl.py`의 고정 런타임, 보상 유지 |
| 2 | `reward_transfer_2_6/.../best.zip` | 현재 팀 공통 `MilestoneEnv` 관측 + 개인 점프 보상, 3행동/6프레임 |
| 4 | 팀 기존 보존 `baseline` PPO | 원래 팀 환경 (개선 작업 보류) |

새 PPO 학습 환경이 필요하면 `stage2_common_bridge.core.make_env(stage)`를 사용할 수 있습니다.
1·3·4·5는 원래 `rl.milestone_env.MilestoneEnv`, 2는 원래 팀 환경을 그대로 감싸는 `RewardOnlyEnv`입니다. PPO 보상은 기존 팀 구간 보상에 중복으로 **추가**하는 대신 개인 보상으로 **교체**하며, 성공/실패 기본 보상은 동일합니다. 입력 크기·행동·물리 규격은 동일합니다.

**중요:** 기존 1·3·5 보존 모델을 현재 `rl/` 코드에 직접 로드하지 않습니다. 팀 `tools/tutorial_rl.py`가 소스 해시를 확인한 뒤 당시 고정 런타임에서 실행합니다. 이 경로를 우회하면 동작이 달라질 수 있습니다.

## 검증의 의미

`verify`: 2단계의 임의 행동을 두 환경에 똑같이 입력해 물리틱, 관측, 종결 상태, 물리좌표가 일치하는지 비교합니다. 개인 PPO에 지급되는 보상은 기본종결+개인 보상만인지 체크합니다. 개인 PPO 체크포인트의 공통 관측·행동 네트워크 호환성을 검사합니다. `--archive`로 팀의 모든 과거 모델·소스 아카이브 해시까지 추가 확인합니다.

`evaluate --stage 2`: 동일하게 4개 독립 확률 샘플링 스트림으로 평가합니다. 이전 96%와 동일한 시드이면 비슷한 결과를 예상할 수 있으나 저장소 내용·환경 버전·모델 파일이 같아야 정확히 재현됩니다. 실행 결과는 `tutorial_2/results/stage2_common_bridge/`에 저장됩니다.

**제약:** 이 ZIP은 기존 환경과 체크포인트가 있는 사용자 PC에서 돌리는 패치입니다. 현재 제작 환경에서는 pygame/gymnasium/stable-baselines3가 없어 실제 게임/가중치 로딩은 실행하지 못했습니다. 정적·단위 테스트는 별도로 통과 여부를 기록합니다.
