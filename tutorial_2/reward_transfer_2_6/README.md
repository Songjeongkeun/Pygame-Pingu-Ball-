> **폴더 이동 안내:** 이 문서는 기존 실험 설명을 보존한 것이야. 실제 위치는 프로젝트 루트의 `tutorial_2/reward_transfer_2_6/`이며 모든 명령은 프로젝트 루트에서 `python -m tutorial_2.reward_transfer_2_6.train ...` 형식으로 실행해. 경로가 헷갈리면 `tutorial_2/README.md`를 우선 참고해.

# 튜토리얼 2단계 | 12프레임 순수 PPO → 6프레임 공통 PPO 적응 실험

**목적:** 이미 독립 평가 200판에서 80.5%(161/200) 성공한 2단계 **3행동·팀 공통 관측 PPO** 가중치를 사용해, **팀의 원래 행동 유지 시간 `6프레임`으로 적응시키는 순수 PPO 추가 학습**을 수행한다. 기존 `12프레임` 최고 체크포인트는 수정·삭제하지 않는다. **1·3·5단계와 4단계는 건드리지 않는다.**

이 실험은 **무작위 초기화부터 새로 학습**하는 것이 아니라, **본인 순수 PPO 가중치에서 PPO를 이어서 학습하는 것(warm start)**이다. **BC, DAgger, 교사 행동 모방, 기존 9행동 체크포인트는 사용하지 않는다.** 보고서에는 이 사실을 구분해서 적는다.

## 환경 사전 준비

GitHub 프로젝트 루트에 기존 `tutorial_2/reward_repeat_2_4/`, `tutorial_2_4/`, `rl/` 폴더가 있어야 한다. 현재 버전은 `tutorial_2/reward_transfer_2_6/` 안에 들어 있다. 기존에 사용한 `(venv)`에서 명령을 실행한다. 원본 PPO는 다음 파일을 기본 입력으로 사용한다.

`tutorial_2/results/reward_repeat_2_4/stage_02/personal/repeat_12/seed_42/best.zip`

모든 파일은 **프로젝트 루트에서** 실행한다.

## 1. 물리·보상 정확성 검사

```cmd
python -m tutorial_2.reward_transfer_2_6.train smoke --macro-steps 300
```

한 번의 오른쪽 12프레임과 두 번의 오른쪽 6프레임이 같은 물리 위치·속도·종료·누적 보상으로 계산되는지 직접 확인한다. 동작이 다르면 학습하지 말고 오류를 확인한다.

## 2. 기존 최고 12프레임 정책을 변경 없이 6프레임에서 평가

```cmd
python -m tutorial_2.reward_transfer_2_6.train baseline --trials 200 --eval-seed 120000
```

동일한 PPO 가중치를 12프레임/6프레임에서 동일 평가 시드로 비교한다. `tutorial_2/results/reward_transfer_2_6/.../before_transfer.json`에 저장한다. 모델 가중치는 바뀌지 않는다. 이 단계에서 6프레임 성공률이 낮아도 12프레임 모델이 손상된 것은 아니다.

## 3. 6프레임에서 PPO 미세 조정 (물리 틱 24만)

```cmd
python -m tutorial_2.reward_transfer_2_6.train train --physics-budget 240000 --seed 42
```

- 단계 2, 보상 `personal`, **공통 관측**, 행동 3개, **6프레임 고정**.
- 초기값: 12프레임 `best.zip`의 **신경망과 PPO 옵티마이저 가중치**. 학습 궤적은 새 6프레임 게임에서 온정책(on-policy)으로 수집.
- 신경망·PPO: **팀의 `MilestonePPO` 그대로**. Gamma=1, GAE=0.98, n_steps=128, batch_size=256, 5 epochs, clip=0.2, target_kl=0.02.
- **2단계 추가 조정:** 학습률 `0.0001`(12프레임 학습 당시 `0.0003`), 엔트로피 `0.01`(당시 `0.05`). 모델 손상을 줄이기 위한 제안이지만 성능 개선은 보장되지 않음.
- 모델 비교·최고 모델 선택은 검증 20판을 기준으로 한다. `best.zip`은 6프레임 **학습 기간**의 최고 모델이며 원본 12프레임 `best.zip`은 보존된다.
- 결과는 새 `tutorial_2/results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42/` 아래에만 저장한다.

## 4. 6프레임 결과 독립 평가 (같은 seed로 비교)

```cmd
python -m tutorial_2.reward_transfer_2_6.train evaluate --trials 200 --eval-seed 120000
```

`baseline`에서 동일 시드로 측정한 적응 전 6프레임 성능과 비교한다. **80.5%는 12프레임에서 측정한 수치이므로 이 평가에서 자동으로 보장되지 않는다.**

## 5. 개선되는 경우에만 물리 틱 48만까지 추가 PPO

```cmd
python -m tutorial_2.reward_transfer_2_6.train train --physics-budget 480000 --seed 42 --resume
```

`--resume`은 이전 **6프레임 적응 실험의 `last.zip`** 에서 이어간다. `best.zip`은 검증 성공률과 중간 목표 비율로 선택한다. 과거의 12프레임 및 팀원 체크포인트는 수정되지 않는다. 또한 이 선택 과정에는 독립 평가 시드 120000이 쓰이지 않는다.

## 필수 주의사항

- `baseline`과 적응 후 `evaluate`에서 **평가 시드 120000을 동일하게** 사용하여 제어 차이와 개선 정도를 비교한다.
- 학습된 6프레임 `best.zip`의 성능이 목표에 못 미칠 수도 있다. 결과로 판단해야 한다.
- 연속 2회 검증에서 성공률 0이며 행동 엔트로피가 매우 낮으면 이전 학습기와 같이 자동 중단한다.
- 전체 게임 환경(`gymnasium`, `pygame`, `stable-baselines3`)이 없는 제작 환경에서는 실제 PPO 게임 학습과 성공률 검증이 불가하다. 원래 사용자 PC에서 `smoke`, `baseline`, `train`, `evaluate` 순서로 테스트한다. 정적/단위 테스트는 포함되어 있다.

## 오류 수정 안내 (2026-11-01)

`AttributeError: 'Namespace' object has no attribute 'stage'`를 수정한 버전이다.
기존 실패 실행에 `config.json`만 남았다면 새 버전으로 파일을 덮어쓴 뒤
**위의 `train` 명령을 `--resume` 없이 다시 실행**하면 된다.
별도 삭제 작업은 필요 없다. 상세 내용은 `ERROR_FIX.md`를 참고한다.
