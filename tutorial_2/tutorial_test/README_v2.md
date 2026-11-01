> **이동 후 실행:** 이 문서는 과거 실험 기록이야. 아래 명령은 프로젝트 루트에서 `tutorial_2.tutorial_test` 패키지로 실행하도록 경로만 맞췄어. 최종 96% 모델은 `tutorial_2/README.md`의 명령을 사용해.

# tutorial_test 보상 정렬 v2 실험 (추가 패치)

이 ZIP은 기존 `tutorial_2/tutorial_test/`에 **새 파일만 추가**합니다. 기존 `personal_rewards.py`, `train_ppo.py`, `results/`와 `rl/`, `tutorial_2_4/`는 변경하지 않습니다.

## 변경 이유
기존 2단계 합산 보상 모델은 100회 평가에서 성공 0회·마일스톤 평균 0·개인 보상 평균 0.485·평균 생존 135초였습니다. 가로 정렬 +0.5가 높이 조건 없이 지급될 수 있는 점을 확인했습니다. 이것이 실패의 유일한 원인이라고 확정할 수는 없습니다.

## 변경한 사항
- 2단계 `target_horizontal` +0.5는 **목표 높이 ±16px와 가로 범위가 동시에 맞았을 때만** 지급됩니다.
- 이미 가로 범위에 들어간 뒤 목표 높이에 도달한 경우도 1회 인정하도록 '동시 정렬'의 진입 상태를 별도로 추적합니다.
- 2단계 `target_height` +0.5, 4단계 리프트 12+12, 공통 보상 100/-25/+10은 그대로입니다.
- 최종 평가 JSON에 `mean_personal_counts`가 추가되어 어떤 보상 조건이 평균 몇 번 발생했는지 확인할 수 있습니다.
- **관측·행동·신경망·기본 PPO 학습 설정과 원본 게임은 변경하지 않습니다.**

## 구성
```
tutorial_test/
  personal_rewards_v2.py
  environment_v2.py
  train_ppo_v2.py
  README_v2.md
  tests/test_personal_rewards_v2.py
```

## 실행 (프로젝트 루트 CMD)
```
python -m unittest discover -s tutorial_2\tutorial_test\tests -p "test_personal_rewards_v2.py" -v
python -m tutorial_2.tutorial_test.train_ppo_v2 smoke --stage 2 --reward combined --actions 100
python -m tutorial_2.tutorial_test.train_ppo_v2 train --stage 2 --reward common --steps 100000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo_v2 train --stage 2 --reward combined --steps 100000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo_v2 compare --stage 2 --seed 42 --trials 500 --eval-seed 60000
```

결과는 **`tutorial_test/results_v2/`**에 저장되어 이전 `tutorial_test/results/`는 보존됩니다. 공통 보상도 v2에서 다시 학습하면 완전히 동일한 실험기/평가 조건으로 비교할 수 있습니다.

이번 수정이 성공률을 개선한다고 보장하지 않습니다. 시드 42 하나로 결론을 내리지 말고 이후 0,1,2 시드 반복을 권장합니다. 또한 팀원 원래 모델을 그대로 이어서 학습하는 실험이 아니라 새 학습입니다.
