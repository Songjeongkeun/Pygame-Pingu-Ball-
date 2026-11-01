> **이동 후 실행:** 이 문서는 과거 실험 기록이야. 아래 명령은 프로젝트 루트에서 `tutorial_2.tutorial_test` 패키지로 실행하도록 경로만 맞췄어. 최종 96% 모델은 `tutorial_2/README.md`의 명령을 사용해.

# tutorial_test v3 — 공통 환경 + 개인 최종 보상 (재검증판)

## 중요: 이번 파일이 무엇을 바꿨는가

**`rl/`, `pingu_ball/`, `tutorial_2_4/`는 절대 덮어쓰지 않습니다.**

이전 v1/v2처럼 2단계인지 4단계인지 직접 확인해서 특정 보상을 지급하는 것이 아니라, **팀원 공통 환경의 마일스톤 시작 발판과 실제 발판 종류**를 읽어 개인 보상을 계산합니다.

| 항목 | common | combined | 출처와 처리 |
|---|---:|---:|---|
| 클리어 | +100 | +100 | 팀원 공통 코드 유지 |
| 실패/시간초과 | -25 | -25 | 팀원 공통 코드 유지 |
| 특수 블록→목표 발판 최초 착지 | +10 | +10 | 팀원 공통 코드 유지, 중복 지급 금지 |
| 점프 후 목표 높이 ±16px 최초 진입 | 0 | +0.5 | 네 최종 2단계 코드에서 **발판ID 기반으로 일반화** |
| 점프 후 목표 x 범위 최초 진입 | 0 | +0.5 | 네 최종 2단계 코드처럼 높이와 **독립** 지급 |
| 상하 이동 발판 최초 착지 | 0 | +12 | 네 최종 4단계 코드, 발판 종류로 판정 |
| 상하 이동 발판 충분히 상승(이동 범위 60% 이상) | 0 | +12 | 고정 `y=334` 제거, 이동 범위에 대한 비율로 치환 |

**일부러 뺀 부분**: 4단계의 오른쪽 진행거리당 +0.045, 특정 결승 발판 상단 도착 +18, 특정 맵의 절대 y좌표, 2·4단계 전용 신경망 입력. 이런 부분은 다른 맵에 그대로 적용할 수 없기 때문입니다.

⚠️ **이 실험은 네 원본 2·4단계 최종 PPO를 그대로 재현하는 것이 아닙니다.** 보상 일부를 팀원의 3행동/공통관측 기반 PPO 환경으로 이식한 *새 실험*입니다. 9행동 `last.pt`를 팀원의 3행동 모델로 그대로 읽어들일 수 없습니다. 개인 보상은 실제 공통 환경의 보상과 더해져 학습기에 `/100` 스케일로 지급됩니다.

## 설치

이 ZIP을 프로젝트 루트 `Pygame-Pingu-Ball-/`에 압축 해제합니다. `tutorial_2/tutorial_test/` 폴더만 추가되고, 기존 v1/v2도 삭제하지 않습니다.

```bat
python -m pip install -r requirements-rl.txt
```

## 1. 시험 (학습 전에 반드시 실행)

```bat
python -m unittest discover -s tutorial_2\tutorial_test\tests -p "test_*_v3.py" -v
python -m tutorial_2.tutorial_test.train_ppo_v3 parity --stage 2 --actions 1000
python -m tutorial_2.tutorial_test.train_ppo_v3 parity --stage 4 --actions 1000
python -m tutorial_2.tutorial_test.train_ppo_v3 smoke --stage 2 --reward combined
python -m tutorial_2.tutorial_test.train_ppo_v3 smoke --stage 4 --reward combined
```

`PARITY PASS`는 공통/합산 환경에서 **동일 행동열의 관측/물리/종료/마일스톤이 정확히 같고, 보상 차이는 개인 보상뿐임**을 뜻합니다. 이 시험이 실패하면 학습하지 마세요.

**안전장치:** `train` 명령 실행 시에도 200회 행동의 동일 물리/관측 검사가 자동 실행됩니다.

## 2. 네 기존 31% 공통 보상 모델 활용 (권장, 2단계)

이미 학습한 공통 환경 2단계 모델이 있으므로, 처음부터 무작위 정책으로 시작하는 대신 같은 정책으로 출발해 개인 보상의 영향을 비교할 수 있습니다. 기존 `results_v2/`의 `best.zip`를 사용합니다.

**A. 공통 보상 그대로 10만 행동 추가학습 (대조군)**

```bat
python -m tutorial_2.tutorial_test.train_ppo_v3 train --stage 2 --reward common --steps 100000 --seed 42 --init-model tutorial_2\tutorial_test\results_v2\tutorial2\common\seed_42\best.zip
```

**B. 같은 공통 모델에서 개인 보상을 추가해 10만 행동 학습 (실험군)**

```bat
python -m tutorial_2.tutorial_test.train_ppo_v3 train --stage 2 --reward combined --steps 100000 --seed 42 --init-model tutorial_2\tutorial_test\results_v2\tutorial2\common\seed_42\best.zip
```

이 두 실험은 **같은 초기 정책 가중치**에서 시작하지만, 추가 PPO 업데이트 때문에 성능이 달라질 수 있습니다. 기존 모델이 좋은지 비교하기 위해 처음 시점의 평가도 별도 체크포인트 후보로 저장합니다. 따라서 최종 저장된 `best.zip`이 기존 정책일 수도 있습니다. `summary.json`의 `selected_initial`을 반드시 확인해 실제 개선인지 판별하세요. 31%를 보장하거나 개선됐다고 주장하지 않습니다.

## 3. 처음부터 학습 (진짜 보상 단독 비교)

2단계:

```bat
python -m tutorial_2.tutorial_test.train_ppo_v3 train --stage 2 --reward common --steps 100000 --seed 43
python -m tutorial_2.tutorial_test.train_ppo_v3 train --stage 2 --reward combined --steps 100000 --seed 43
```

4단계:

```bat
python -m tutorial_2.tutorial_test.train_ppo_v3 train --stage 4 --reward common --steps 100000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo_v3 train --stage 4 --reward combined --steps 100000 --seed 42
```

4단계가 기존에 0%였기 때문에 10만 행동만으로 높은 성공률을 얻는다고 보장할 수 없습니다. 여러 시드와 더 긴 학습을 별도 실험으로 비교해야 합니다. 스텝을 변경하면 기존 폴더를 덮어쓰지 않도록 새로운 시드 또는 `--root`를 설정하세요.

## 4. 저장 모델 재평가

```bat
python -m tutorial_2.tutorial_test.train_ppo_v3 evaluate --folder tutorial_2\tutorial_test\results_v3\stage_02\combined_from_common\seed_42 --trials 500
```

저장 경로 `results_v3/`은 이전 `results/`·`results_v2/`와 독립입니다. `config.json`, `best.zip`, `initial.zip`, `last.zip`, `validations.csv`, `best_validation.json`, `final_evaluation.json`, `summary.json`이 저장됩니다. 학습 당시 코드 SHA-256이 달라지면 `evaluate`가 중단됩니다.

## 5. 실제 검증 범위

파일 제작 환경에서는 numpy/torch만 설치되어 있고 GitHub 게임 전체 소스와 `gymnasium`, `stable-baselines3` 패키지를 실행할 수 없습니다. 그래서 **이 ZIP 제작 시점에는 실제 PPO 10만 스텝 학습을 수행하거나 성공률을 검증하지 못했습니다.** 순수 보상 로직 단위 테스트 15개 통과, 통합 테스트 2개는 의존성 부족으로 건너뛰었습니다. 사용자 환경에서 `parity`와 `smoke`, 학습을 완료해야 실제로 성능이 확인됩니다.

## 6. 기존 실험 0%의 해석

팀원 공통 환경에는 3개 행동과 단계 공통 관측이 있고, 네 원본 최종 PPO에는 9개 행동과 단계 전용 관측이 있습니다. 따라서 네 최종 100%/99.8%가 개인 보상 하나만의 효과라고 단정할 수 없습니다. 기존 v2의 2단계 합산 보상은 100k에서 0%와 135초의 긴 대기가 관찰되었지만, **원인이 정렬 보상의 버그라고 확정되지는 않았습니다.** 이번 버전은 보상 이식의 정확성 검증, 결과 분리, 가중치 이어학습을 추가했습니다.
