> **폴더 이동 안내:** 이 문서는 기존 실험 설명을 보존한 것이야. 실제 위치는 프로젝트 루트의 `tutorial_2/reward_repeat_2_4/`이며 모든 명령은 프로젝트 루트에서 `python -m tutorial_2.reward_repeat_2_4.train ...` 형식으로 실행해. 경로가 헷갈리면 `tutorial_2/README.md`를 우선 참고해.

# 튜토리얼 2·4 | 공통 3행동 PPO에서 행동 유지 시간 비교 (6프레임 vs 12프레임)

## 무엇을 확인하는 실험인가?

기존 2단계 6프레임 순수 PPO는 첫 점프 블록 착지율 91%, 평균 목표 발판 착지 0.73회(독립평가 100회)까지 개선되었지만, 성공률은 0%였다. 이동 방향을 더 오래 유지하면 2번째 점프 블록까지 넘어가는 데 도움이 되는지 비교한다.

**게임 물리, 지형, 팀 공통 관측, 행동 개수 3개, 팀 특징 추출기, PPO 하이퍼파라미터, 2·4단계 개인 보상은 변경하지 않는다.** `MilestoneEnv(..., repeat=6)` 대신 `repeat=12`를 적용하는 것이 유일한 환경 조작이다. 엔트로피 계수는 앞서 실시한 **hold=0.05** 탐색 실험과 동일하게 고정한다. 다른 옵션(첫 점프 블록 추가 보상 포함)은 사용하지 않는다.

- 1·3·5단계 학습 코드, 가중치, 데이터는 수정하지 않는다.
- BC, DAgger, 교사 정책·데모·사전학습 모델을 사용하지 않는다. 2·4단계는 무작위 초기화한 팀 순수 PPO로 시작한다. `--resume`은 **같은 실험에서 방금 학습한 자기 자신의 PPO 체크포인트**만 이어받는다.
- 2단계 +10 목표 발판 착지, +0.5 목표 높이, +0.5 가로 정렬 보상 그대로; 성공 +100/실패 -25.
- 4단계 리프트 착지·상승·상단 발판·접근 보상 그대로; 성공 +100/실패 -25.
- `spring2`, `middle`, `upper`는 **관측이나 보상에 추가하지 않는 로그 지표**다.

## 설치

프로젝트 최상위 (`Pygame-Pingu-Ball-`) 폴더에 `tutorial_2/`를 두며, `tutorial_2/reward_repeat_2_4/` 안에 들어 있다. 기존 `reward_explore_2_4`, `reward_anneal_2_4`, `tutorial_2_4`, `rl` 폴더를 덮어쓰지 않는다. 기존 게임·학습 가상환경에서 실행한다.

## 1. 먼저 검증 (중요)

```cmd
python -m tutorial_2.reward_repeat_2_4.train smoke --stage 2 --repeat 12 --decisions 1000
```

`[SMOKE PASS]`가 나오면 **동일한 12프레임 입력 시 팀 원본 환경과 새 개인 보상 환경의 게임 물리·관측·종료 판정이 완전히 일치**한 것이다. 보상 값만 달라진다. 오류가 나면 학습시키지 말고 오류 로그를 확인한다.

## 2. 2단계 12프레임, 24만 물리 틱까지만 먼저 학습

```cmd
python -m tutorial_2.reward_repeat_2_4.train train --stage 2 --repeat 12 --reward personal --physics-budget 240000 --seed 42
```

- 모델은 처음부터 PPO로 학습한다 (이전 6프레임 가중치 불러오지 않음).
- 팀 3행동(왼쪽·대기·오른쪽) 및 공통 관측 그대로.
- `learning_rate=.0003`, `gamma=1`, `GAE=.98`, `n_steps=128`, `batch_size=256`, `epochs=5`, `clip=.2`, `target_kl=.02`, `ent_coef=.05`.
- 평가 간격: **실제 물리 틱 6만** 정도마다 검증 20판.
- 출력: `physics`(훈련 실제 틱), `spring1`, `middle`, `spring2`, `upper`, `max_x`, `policy_H`, 평균 보상, 성공률.
- 학습량은 행동 결정 횟수가 아니라 각 환경의 `info['tick_delta']`를 합한 **실제 물리 틱**으로 종료한다 (마지막 묶음에서 목표치를 조금 초과할 수 있음).

체크포인트와 로그:

```text
tutorial_2/results/reward_repeat_2_4/stage_02/personal/repeat_12/seed_42/
  config.json
  validation_history.jsonl
  best.zip
  last.zip
  latest.json
  best_validation.json
  summary.json
```

`best.zip`은 검증 성공률·고급 구간 도달률 기준 최선 모델, `last.zip`은 마지막 PPO 모델이다.

## 3. 2단계가 개선되었다면 총 48만 물리 틱까지 연장

```cmd
python -m tutorial_2.reward_repeat_2_4.train train --stage 2 --repeat 12 --reward personal --physics-budget 480000 --seed 42 --resume
```

이전 6프레임 실험의 `80,000 결정 × 최대 6프레임 = 약 480,000 물리 틱`과 유사한 총 물리 시뮬레이션량이다. 앞선 6프레임 실험은 실제 틱을 기록하지 않았으므로 **엄밀한 등량 검증을 하려면 아래 6프레임 대조군도 새로 실행해야 한다.** 또한 틱 수가 같아도 PPO 업데이트 횟수(결정 횟수)가 달라 학습 연산량은 같지 않다.

## 4. 독립 평가 100판

```cmd
python -m tutorial_2.reward_repeat_2_4.train evaluate --stage 2 --repeat 12 --reward personal --model tutorial_2/results/reward_repeat_2_4/stage_02/personal/repeat_12/seed_42/best.zip --trials 100 --seed 70000
```

## 5. 선택: 공정한 6프레임 대조군 (동일 새 학습기, 물리 틱 동일)

```cmd
python -m tutorial_2.reward_repeat_2_4.train train --stage 2 --repeat 6 --reward personal --physics-budget 480000 --seed 42
```

```cmd
python -m tutorial_2.reward_repeat_2_4.train evaluate --stage 2 --repeat 6 --reward personal --model tutorial_2/results/reward_repeat_2_4/stage_02/personal/repeat_6/seed_42/best.zip --trials 100 --seed 70000
```

6프레임과 12프레임을 둘 다 이 스크립트로 **같은 시드, 같은 물리 틱 예산, 같은 개인 보상, 같은 PPO 설정, 같은 독립 평가**로 비교하면 행동 유지 시간의 영향을 더 분명히 볼 수 있다. 단, 의사결정 수 및 그에 따른 PPO 업데이트 수가 서로 달라진다는 점을 비교 보고서에 명시한다.

## 4단계는 2단계 결과를 본 뒤

```cmd
python -m tutorial_2.reward_repeat_2_4.train smoke --stage 4 --repeat 12 --decisions 1000
python -m tutorial_2.reward_repeat_2_4.train train --stage 4 --repeat 12 --reward personal --physics-budget 240000 --seed 42
```

## 중요한 주의사항

- 이전 6프레임 `best.zip`에서 12프레임을 시작하지 않는다. **환경 동역학이 달라진 비교 실험이므로 새 PPO로 학습**한다.
- 최종 성공률이 0%일 수도 있다. 이번 파일은 **행동 유지 시간이 병목인지 검증하는 실험**이며 성공률 보장 수단이 아니다.
- 본 도구 제작 환경에는 `pygame`, `gymnasium`, `stable_baselines3`가 없어 실제 게임에서 전체 PPO 학습은 실행하지 못했다. 정적 분석과 의존성 대체 단위 테스트 8개를 통과했으며, 실제 환경 검증은 위 `smoke`가 담당한다.
- `python -m pytest -q tutorial_2/reward_repeat_2_4/tests`는 설치된 `pytest`가 있으면 추가적으로 실행할 수 있다.
