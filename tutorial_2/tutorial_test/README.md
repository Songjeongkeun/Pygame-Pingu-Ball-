> **이동 후 실행:** 이 문서는 과거 실험 기록이야. 아래 명령은 프로젝트 루트에서 `tutorial_2.tutorial_test` 패키지로 실행하도록 경로만 맞췄어. 최종 96% 모델은 `tutorial_2/README.md`의 명령을 사용해.

# 튜토리얼 2·4단계 공통 환경 + 개인 보상 비교

팀원이 만든 `rl/milestone_env.py`의 관측·행동·공통 마일스톤과 `rl/milestone_models.py`의 PPO 신경망을 재사용한다. 기존 `rl/`, `tutorial_2_4/`, `pingu_ball/` 파일과 학습된 `.pt` 모델은 수정하지 않는다. 본 폴더의 새 PPO는 **기존 모델을 이어 학습하는 것이 아니라 처음부터 학습**하는 독립 실험이다.

## 파일 구성

| 파일·폴더 | 설명 |
|---|---|
| `environment.py` | 팀 공통 강화학습 환경을 재사용하면서 보상만 선택적으로 합치는 환경 |
| `personal_rewards.py` | 2·4단계의 재사용 가능한 개인 보상 계산 및 중복 방지 |
| `train_ppo.py` | 환경 확인, PPO 신규 학습, 저장 모델 평가, 공통/합산 보상 비교 |
| `tests/` | 보상 조건 및 중복 지급 검사 |
| `results/` | 학습 시 자동 생성되는 결과 폴더. ZIP에는 미포함 |

## 보상 규칙

`--reward common`은 팀원 공통 보상만, `--reward combined`는 동일한 공통 보상에 아래 개인 보상을 더한다. 두 실험 모두 **같은 관측값·행동 3개·신경망·PPO 설정**을 사용한다.

| 구분 | 조건 | 점수 |
|---|---|---:|
| 공통 | 이글루 도착 | +100 |
| 공통 | 실패 또는 180초 초과 | -25 |
| 공통 | 특수 블록 착지 후 지정된 목표 발판 그룹 최초 착지 | 구간당 +10 |
| 개인 2단계 | 점프 블록 착지 후 목표 발판 높이 ±16px에 새롭게 진입 | 출발 블록당 +0.5 |
| 개인 2단계 | 점프 블록 착지 후 목표 발판의 가로 범위에 새롭게 진입 | 출발 블록당 +0.5 |
| 개인 4단계 | 상하 이동 블록에 처음 착지 | 해당 장치당 +12 |
| 개인 4단계 | 착지 직후 리프트가 이동 범위의 상단 40%에 있고 공이 가까운 경우 | 해당 장치당 +12 |
| 공통 | 일반 이동·대기·생존 | 0 |

2단계 개인 정렬 보상은 점프 블록을 실제로 밟은 뒤 3초 내에만 활성화되고, 다른 발판에 착지하면 해당 시도가 끝난다. 동일 점프 블록에서는 높이·가로 보상을 각 한 번만 지급한다. 4단계 상단 활용은 절대 높이 `y=334` 대신 리프트 전체 이동 거리의 60% 이상을 올라온 상태로 판정한다. 최초 착지 이후 1.05초 이내이고 공이 리프트 근처에 있어야 한다. 개인 보상은 각 조건을 한 판에서 한 번만 지급한다.

**제외한 튜토리얼 전용 보상:** 4단계 특정 `finish-` 발판에 대한 별도 +18, 맵 오른쪽으로 이동한 거리당 +0.045, 고정 높이 좌표 334. 2단계의 마일스톤 +10은 이미 공통 환경에서 지급하므로 중복 추가하지 않는다. 4단계의 리프트→목표 발판 +10도 공통 환경에서 한 번만 지급한다.

공통 환경의 출발·목표 발판 그룹은 현재 팀원 코드의 `rl/milestone_env.py`에 정의된 **튜토리얼 2·4단계 구간 정보**를 사용한다. 별도의 튜토리얼 전용 추가 보상을 없앤 것이지, 새로운 일반 맵 10개까지 자동으로 지원하는 것은 아니다.

## PPO 공통 설정

| 항목 | 설정값 |
|---|---|
| 알고리즘 | PPO, Stable-Baselines3, CPU |
| 관측·모델 | 팀 공통 `MilestoneEnv` 관측 구조 및 `MilestoneExtractor` |
| 행동 | 왼쪽·무입력·오른쪽 3개, 6물리 틱 유지(0.05초) |
| 학습 시작 | 무작위 초기 가중치, 행동별 초기 확률 동일 |
| 학습률 | 0.0003 |
| 감가율(Gamma) | 1.0 |
| GAE Lambda | 0.98 |
| 병렬 환경 수 / 환경별 수집 | 4개 / 128스텝 |
| 미니배치 / PPO 갱신 | 256 / 5회 |
| Clip Range / Target KL | 0.2 / 0.02 |
| 가치 손실 계수 / 경사 제한 | 0.5 / 0.5 |
| 엔트로피 | 처음 20%까지 0.02, 80%에서 0.005, 종료 시 0.001 |
| 학습용 보상 | 원점수 ÷ 100 (보고서는 원점수) |
| 중간 평가 | 기본 2만 행동 선택마다 20회, 확률적 정책 |
| 최종 평가 | 저장된 최고 검증 모델, 기본 100회 별도 시험 |
| 권장 비교 | 같은 시드·학습 스텝·평가 시드로 `common` 대 `combined` 비교 |

## 설치 및 실행

프로젝트 최상위 폴더(`pingu_ball/`, `rl/`, `tutorial_2/tutorial_test/`가 나란히 있는 곳)에서 가상환경을 활성화한 후 실행한다.

```bat
python -m pip install -r requirements-rl.txt
python -m unittest discover -s tutorial_2\tutorial_test\tests -v
python -m tutorial_2.tutorial_test.train_ppo smoke --stage 2 --reward combined --actions 100
python -m tutorial_2.tutorial_test.train_ppo smoke --stage 4 --reward combined --actions 100
```

먼저 10만 행동 선택으로 2단계의 두 보상 조건을 비교하려면 다음과 같이 실행한다. 두 학습 모두 새로 시작한다.

```bat
python -m tutorial_2.tutorial_test.train_ppo train --stage 2 --reward common --steps 100000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo train --stage 2 --reward combined --steps 100000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo compare --stage 2 --seed 42 --trials 500 --eval-seed 60000
```

4단계는 `--stage 4`로 바꾸어 같은 방식으로 실행하면 된다. 최종 100만 행동 선택 비교를 원하면 두 학습 모두 `--steps 1000000`을 사용한다. 같은 저장 폴더에 재학습을 시도하면 덮어쓰지 않고 중단된다. 새 실험은 `--root` 또는 `--seed`를 다르게 지정한다.

```bat
python -m tutorial_2.tutorial_test.train_ppo train --stage 4 --reward common --steps 1000000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo train --stage 4 --reward combined --steps 1000000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo compare --stage 4 --seed 42 --trials 500 --eval-seed 60000
```

## 결과 파일

학습 후 `tutorial_test/results/tutorial2/common/seed_42/`와 `tutorial_test/results/tutorial2/combined/seed_42/`처럼 구분해 저장한다. 4단계도 동일하다.

| 파일 | 의미 |
|---|---|
| `initial.zip` | 학습 시작 전 초기 정책 |
| `best.zip` | 중간 검증에서 성능이 가장 좋았던 정책 |
| `last.zip` | 마지막 학습 정책 |
| `config.json` | 학습 설정, 원본 환경·보상 코드 해시 |
| `evaluations.csv` | 학습 스텝별 평가 성공률 및 학습 지표 |
| `best_validation.json` | 최고 정책 선택 기준과 중간 평가 결과 |
| `final_evaluation.json` | 선택된 최고 모델의 별도 최종 시험 |
| `summary.json` | 학습 횟수, 최종 시험, 코드 변경 검사 |
| `progress.csv` | Stable-Baselines3의 학습 로그 |

현재 포함된 것은 **실행 코드 및 테스트**다. 새 공통 환경 + 개인 보상 모델은 아직 학습되지 않았으며 성공률도 아직 없다. 기존 2단계 100%, 4단계 99.8%는 별도 전용 환경에서 얻은 결과이므로 이 실험의 점수로 표시하지 않는다.
